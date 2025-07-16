import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def rho_idx(mask_eps: float, D: int, delta_idx: float = 1e-5) -> float:
    """
    High-probability zCDP rho for the Gumbel-TopK mask.
    """
    return 8.0 * (mask_eps ** 2) * math.log(D / delta_idx)

def rho_val(K: int, sigma_N2: float) -> float:
    """
    zCDP rho for releasing K Gaussian-perturbed values.
    """
    return K / (2.0 * sigma_N2)

def eps_dp_from_rho(rho: float, delta_final: float = 1e-5) -> float:
    """
    Convert zCDP rho to (ε, δ_final)-DP.
    """
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta_final))

class SplitNN(nn.Module):
    def __init__(self, base_model, split_logic, defense_module=None):
        super().__init__()

        # --- Model Splitting Logic ---
        if 'vgg' in base_model.__class__.__name__.lower():
            # VGG-specific split
            all_features = list(base_model.features.children())
            client_layers = all_features[:split_logic]
            server_feature_layers = all_features[split_logic:]
            
            # The server part includes the rest of features, the avgpool, and the classifier
            server_modules = [nn.Sequential(*server_feature_layers)]
            server_modules.append(base_model.avgpool)
            server_modules.append(nn.Flatten(1)) # <<< THE FIX IS HERE
            server_modules.append(base_model.classifier)
            
        else:
            # General ResNet/DenseNet split
            all_layers = list(base_model.children())
            # Determine split index based on layer name (e.g., 'layer2')
            split_idx = [i for i, (name, _) in enumerate(base_model.named_children()) if name == split_logic]
            if not split_idx:
                raise ValueError(f"Split layer '{split_logic}' not found in model.")
            split_idx = split_idx[0] + 1

            client_layers = all_layers[:split_idx]
            server_layers = all_layers[split_idx:]
            
            # Reconstruct server modules, inserting Flatten where needed
            server_modules = []
            for module in server_layers:
                server_modules.append(module)
                # If we just added a pooling layer, the next step in the original forward pass
                # is always a flatten before the fully-connected layer.
                if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                    server_modules.append(nn.Flatten(1)) # <<< THE FIX IS HERE

        # --- Assemble Client and Server Models ---
        self.client_model = nn.Sequential(*client_layers)
        if defense_module:
            self.client_model.add_module("defense", defense_module)
        
        self.server_model = nn.Sequential(*server_modules)

    def forward(self, x, return_intermediate=False):
        intermediate = self.client_model(x)
        output = self.server_model(intermediate)
        if return_intermediate:
            return intermediate, output
        return output


def _apply_unstructured_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    """Identifies the top-k activations in each sample and zeroes out the rest."""
    if k <= 0: return torch.zeros_like(x)
    flat_x = x.view(x.size(0), -1)
    kth_vals, _ = flat_x.kthvalue(flat_x.size(1) - k + 1, dim=1, keepdim=True)
    mask = (flat_x >= kth_vals).float()
    return (flat_x * mask).view(x.shape), mask



class RandomizedTopKBaseline(nn.Module):
    """Non-parametric baseline that applies Randomized Top-K directly."""
    def __init__(self, top_p, mask_epsilon, **kwargs):
        super().__init__()
        self.top_p = top_p
        self.epsilon = mask_epsilon
    def encode(self, x):
        pre_acts = x
        k = max(1, int(self.top_p * pre_acts.numel() / pre_acts.size(0)))
        std_dev = pre_acts.view(pre_acts.size(0), -1).std(dim=1, keepdim=True)
        gumbel_scale = 1.0 / (self.epsilon + 1e-6)
        beta = (std_dev * gumbel_scale).view(-1, 1, 1, 1)
        gumbel_noise = torch.distributions.Gumbel(0.0, 1.0).sample(pre_acts.shape).to(pre_acts.device)
        acts = F.relu(pre_acts + beta * gumbel_noise)
        return _apply_unstructured_topk(acts, k=k)
    def decode(self, sparse_code): return sparse_code
    def forward(self, x):
        sparse_code = self.encode(x)
        return sparse_code, sparse_code

class ConvTopKSAE(nn.Module):
    """Convolutional Top-K Sparse Autoencoder (SAE)."""
    def __init__(self, num_channels, hidden_channel_scale, top_p, mask_epsilon,normalize_decoder=True, **kwargs):
        super().__init__()
        self.num_channels, self.hidden_channels = num_channels, num_channels * hidden_channel_scale
        self.top_p, self.normalize_decoder = top_p, normalize_decoder
        self.W_enc = nn.Parameter(torch.empty(self.hidden_channels, self.num_channels, 1, 1))
        self.b_enc = nn.Parameter(torch.zeros(self.hidden_channels))
        self.b_dec = nn.Parameter(torch.zeros(self.num_channels))
        self.epsilon = mask_epsilon
        self.std_list = []
        nn.init.xavier_uniform_(self.W_enc)
    @property
    def W_dec(self):
        w_dec = self.W_enc.transpose(0, 1)
        return F.normalize(w_dec, p=2, dim=1) if self.normalize_decoder else w_dec
    def encode(self, x):
        x_centered = x - self.b_dec.view(1, -1, 1, 1)
        pre_acts = F.conv2d(x_centered, self.W_enc, self.b_enc)
        

            
        k = max(1, int(self.top_p * pre_acts.numel() / pre_acts.size(0)))
        std_dev = pre_acts.view(pre_acts.size(0), -1).std(dim=1, keepdim=True)
        self.std_list.append(torch.mean(std_dev).detach().item())
        gumbel_scale = 1.0 / self.epsilon + 1e-6
        beta = (std_dev * gumbel_scale).view(-1, 1, 1, 1)
        print(beta.max().item())
        gumbel_noise = torch.distributions.Gumbel(0.0, 1.0).sample(pre_acts.shape).to(pre_acts.device)
        acts = F.relu(pre_acts + beta * gumbel_noise)
        acts, mask = _apply_unstructured_topk(acts, k=k)
        return acts
    def decode(self, sparse_code):
        return F.conv2d(sparse_code, self.W_dec) + self.b_dec.view(1, -1, 1, 1)
    def forward(self, x):
        sparse_code = self.encode(x)
        return self.decode(sparse_code), sparse_code

class SAE_Wrapper(nn.Module):
    """A simple wrapper to make the SAE forward pass compatible with nn.Sequential."""
    def __init__(self, sae_module):
        super().__init__()
        self.sae = sae_module
    def forward(self, x):
        recon, _ = self.sae(x)
        return recon

def calibrate_gaussian_noise_for_value_channel(
    sparse_code, 
    value_channel_leakage_budget_bits,
    eps=1e-9
):
    """
    Calculates the required Gaussian noise `σ_N` to meet a specific
    value-channel privacy leakage budget.

    Args:
        sparse_code (torch.Tensor): The sparse feature tensor `z'` after the
                                    Randomized Top-K operation.
        value_channel_leakage_budget_bits (float): The target maximum leakage
                                                   in bits for the value channel.
        eps (float): A small value for numerical stability.

    Returns:
        float: The calculated standard deviation `σ_N` for the Gaussian noise.
    """
    # We don't need gradients for this defense mechanism
    with torch.no_grad():
        # Step 1: Measure the signal variance `σ_z'²`
        # We only consider the non-zero elements.
        non_zero_values = sparse_code[sparse_code != 0]
        
        if non_zero_values.numel() <= 1:
            # Not enough data to calculate variance, or no signal to protect.
            # Returning a small default noise is safe.
            return 1e-4

        sigma_z_prime_squared = torch.var(non_zero_values).item()

        # Get k, the number of non-zero elements per sample in the batch
        # We assume k is constant across the batch
        k = non_zero_values.numel() / sparse_code.size(0)

        # Handle edge cases for the budget
        if value_channel_leakage_budget_bits <= 0:
            # A budget of <= 0 bits requires infinite noise.
            return float('inf')
        
        # Step 2: Calculate the denominator from the inverse formula
        exponent = (2.0 * value_channel_leakage_budget_bits) / k
        denominator = math.pow(2, exponent) - 1
        
        if denominator <= eps:
            # This happens if the budget is extremely small, also requires infinite noise.
            return float('inf')

        # Step 3: Calculate the required noise variance and standard deviation
        noise_variance = sigma_z_prime_squared / denominator
        noise_stddev = math.sqrt(noise_variance)

    return noise_stddev