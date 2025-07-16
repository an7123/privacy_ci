# main.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import psutil
import threading
from torchvision.datasets import ImageFolder
# Import all defense modules and helper functions
from defenses import (
    SplitNN, RandomizedTopKBaseline, ConvTopKSAE, SAE_Wrapper, calibrate_gaussian_noise_for_value_channel
)
from nncodec.tensor import encode, decode

# ==============================================================================
# == Imports for GAN Inversion Attack
# ==============================================================================
import torch.nn.functional as F
from torch.autograd import grad
# Assuming GMI and torch_ssim are installed and in the python path
# pip install torch-ssim
from GMI.utils import save_tensor_images
import architectures_torch as architectures
from GMI.discri import DGWGAN32
from torch_ssim import SSIM
import numpy as np

# ==============================================================================
# == 0. UTILITY & MONITORING FUNCTIONS
# ==============================================================================
peak_cpu_memory = 0
stop_monitoring = False

def monitor_cpu_memory():
    global peak_cpu_memory, stop_monitoring
    process = psutil.Process(os.getpid())
    stop_monitoring = False
    while not stop_monitoring:
        try:
            mem_info = process.memory_info().rss
            if mem_info > peak_cpu_memory: peak_cpu_memory = mem_info
        except psutil.NoSuchProcess: break
        time.sleep(0.5)

# ==============================================================================
# == 1. MAIN EXECUTION SCRIPT
# ==============================================================================
def main():
    monitor_thread = threading.Thread(target=monitor_cpu_memory, daemon=True)
    monitor_thread.start()
    start_time = time.monotonic()

    parser = argparse.ArgumentParser(description='SplitNN Privacy Defenses Framework')
    
    # --- Universal Arguments ---
    uni_group = parser.add_argument_group('Universal Configuration')
    uni_group.add_argument('--method', type=str, default='baseline', 
                           choices=['baseline', 'blockdp','bottleneck', 'dropout', 'laplace_noise', 'nopeek', 'noisy_arl', 'strong_arl', 'patrol', 'sae', 'rand_topk'],
                           help='Privacy defense method to use.')
    uni_group.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use.')
    uni_group.add_argument('--data-root', type=str, default='./', help='Dataset Path.')
    uni_group.add_argument('--arch', type=str, default='resnet18', help='Model architecture (e.g., resnet18, vgg11).')
    uni_group.add_argument('--split-layer', type=str, default='layer2', help='Name of the layer to split ResNet-like models at.')
    uni_group.add_argument('--cutting-layer', type=int, default=8, help='Layer index to split VGG-like models (within `features`).')
    uni_group.add_argument('--epochs', type=int, default=40, help='Number of training epochs.')
    uni_group.add_argument('--batch-size', type=int, default=128, help='Input batch size for training.')
    uni_group.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    uni_group.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    uni_group.add_argument('--use-nnc', action='store_true', help="Run Compression on intermediate features.")
    # --- Arguments for Specific Defenses ---
    defense_group = parser.add_argument_group('Defense Specific Arguments')
    defense_group.add_argument('--dropout-rate', type=float, default=0.5, help='[Dropout] p value for dropout.')
    defense_group.add_argument('--laplace-scale', type=float, default=0.1, help='[Laplace] Scale parameter (b) for Laplace noise.')
    defense_group.add_argument('--bottleneck-size', type=int, default=16, help='[Bottleneck] Size of the bottleneck layer.')
    defense_group.add_argument('--dcor-weight', type=float, default=0.2, help='Weight of decorrelation loss for NoPeek.')
    defense_group.add_argument('--value-channel-budget', type=int, default=400, help='Weight of decorrelation loss for NoPeek.')
    defense_group.add_argument('--noise_warm_epochs', type=int, default=10, help='Warump Epochs for Noise.')
    # --- Arguments for SAE / Rand_TopK Methods ---
    sae_group = parser.add_argument_group('SAE / Robust E2E Specific Arguments')
    sae_group.add_argument('--hidden-scale', type=int, default=2, help='[SAE] SAE hidden layer channel scale.')
    sae_group.add_argument('--top-p', type=float, default=0.1, help='[SAE/TopK] Sparsity: top P-percent of activations to keep.')
    sae_group.add_argument('--task-loss-weight', type=float, default=1.0, help='[SAE/TopK] Weight for the main classification task loss.')
    sae_group.add_argument('--recon-loss-weight', type=float, default=0.1, help='[SAE] Weight for the SAE reconstruction loss.')
    sae_group.add_argument('--adversarial-training', action='store_true', help="[SAE/TopK] Enable PGD adversarial training.")
    sae_group.add_argument('--adv-epsilon', type=float, default=0.015, help='[SAE/TopK] PGD attack epsilon.')
    sae_group.add_argument('--noise-aware-training', action='store_true', help="[SAE/TopK] Enable noise-aware training for sparse codes.")
    sae_group.add_argument('--noise-sigma', type=float, default=0.8, help='[SAE/TopK] Std deviation for noise injection.')
    sae_group.add_argument('--client-ckpt', type=str, default='checkpoints/client.pth', help='[SAE/TopK/etc.] Path to save client model checkpoint.')
    sae_group.add_argument('--server-ckpt', type=str, default='checkpoints/server.pth', help='[SAE/TopK/etc.] Path to save server model checkpoint.')
    sae_group.add_argument('--sae-ckpt', type=str, default='checkpoints/sae.pth', help='[SAE/TopK] Path to save SAE/Defense model checkpoint.')
    sae_group.add_argument('--mask-epsilon', type=float, default=10.0, help='Epsilon Value for Sparsification mask privacy')

    # --- Arguments for GAN Inversion Attack ---
    gan_group = parser.add_argument_group('GAN Inversion Attack')
    gan_group.add_argument('--run-attack', action='store_true', help='Run GAN inversion attack after training.')
    gan_group.add_argument('--gan-epochs', type=int, default=100, help='[GAN] Number of training epochs for the attack GAN.')
    gan_group.add_argument('--gan-lr', type=float, default=2e-4, help='[GAN] Learning rate for Generator and Discriminator.')
    gan_group.add_argument('--mse-weight', type=float, default=100.0, help='[GAN] Weight for the reconstruction (MSE) loss in G.')
    gan_group.add_argument('--gan-noise-strength', type=float, default=0.1, help='[GAN] Sigma for noise added to activations for GAN training.')
    gan_group.add_argument('--gan-critic-steps', type=int, default=5, help='[GAN] Train G every n_critic steps.')
    gan_group.add_argument('--gan-output-dir', default='./gan_inversion_results', help='[GAN] Directory to save logs, models, and images.')
    
    
    


    args = parser.parse_args()
    os.makedirs('checkpoints', exist_ok=True)
    
    print("--- Configuration ---")
    for key, val in vars(args).items(): print(f"{key}: {val}")
    print("---------------------\n")
    
    if args.use_nnc:
        nnc_config = \
            {"sparsity": 0.0,  # Target unstructured sparsity (0.0 to 1.0)
             "qp": -32,  # Quantization parameter
             "struct_spars_factor": 0.0,  # Gain factor for structured sparsification
             "results": f'/home/hoefler/privacy_ci/final_version/bitstream',  # Directory to save the bitstream
             "tensor_path": None, # path to tensor if saved to disk, e.g. /tensor.pt or ./tensor.npy
             "job_identifier": "SplitXference3000",  # Unique identifier for the current job
             "bitdepth": 4,  # Integer quantization bit width (e.g., 4 or 8), or None for qp-based quantization
             "use_dq": True,  # Whether to use dependent scalar / Trellis-coded quantization
             "approx_method": "uniform",  # Approximation method: 'uniform' or 'codebook'
             "opt_qp": False,  # Whether to optimize QP per tensor based on tensor-size
             "row_skipping": True,  # Whether to use row skipping
             "tca": False,  # Whether to apply tensor core acceleration
             "verbose": False,  # Verbose output flag
             "tensor_id": '0', # identifier for coded tensor in the bitstream (default: '0')
             "quantize_only": False, # returns only quantized tensor instead of NNC bitstream
            }

    # --- Setup Device and Data ---
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:1")
    
    print(f"Using device: {device}")
    #if use_cuda: torch.cuda.reset_peak_memory_stats(device)

    if args.dataset == 'tiny_imagenet':
        
        input_size = (64,64)
        
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
        val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = ImageFolder(os.path.join(args.data_root, 'train'), transform=train_transform)
        test_dataset = ImageFolder(os.path.join(args.data_root, 'val'), transform=val_transform)
        num_classes = len(train_dataset.classes)
        print(f"  - Found {len(train_dataset)} training images belonging to {num_classes} classes.")
        print(f"  - Found {len(test_dataset)} validation images.")
    # Add other dataset loaders here if needed (e.g., CIFAR10)
    
    elif args.dataset == 'facescrub':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Note: The input size must match what the model expects (e.g., 64x64)
        input_size = (48,48)

        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        val_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # Use ImageFolder to load the data
        train_dataset = ImageFolder(root=os.path.join(args.data_root, 'train'), transform=train_transform)
        test_dataset = ImageFolder(root=os.path.join(args.data_root, 'val'), transform=val_transform)
        num_classes = len(train_dataset.classes)
        print(args.dataset, num_classes)
    else:
        # Default to CIFAR10 if not tiny_imagenet
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform)
        num_classes = 10
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    _, input_channels, img_size, _ = next(iter(train_loader))[0].shape

    # --- Model, Optimizer, and Loss Setup ---
    if args.arch == 'resnet18':
        base_model = models.__dict__[args.arch](weights='DEFAULT' if 'resnet' in args.arch else None) # Use pretrained for ResNet
    # if args.arch == 'vgg11':
    #     base_model = vgg11_bn_sgm(
    #     cutting_layer=args.cutting_layer,
    #     logger=None,
    #     num_class=num_classes,
    #     adds_bottleneck=True, # The 'sgm' version implies a bottleneck
    #     bottleneck_option=args.bottleneck_option,
    #     feature_size=args.feature_size
    # ).to(device)
    
    # Adapt final layer for the specific dataset
    if 'resnet' in args.arch or 'resnext' in args.arch:
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Linear(num_ftrs, num_classes)
    # elif 'vgg' in args.arch:
    #     num_ftrs = base_model.classifier[-1].in_features
    #     base_model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        
    if args.dataset == 'tiny_imagenet':
        # Load a pre-trained model on Tiny-ImageNet if available for better starting point
        try:
            base_model.load_state_dict(torch.load('/checkpoints/tiny_imagenet_resnet18.pth'))
            print("Loaded pre-trained weights for Tiny-ImageNet.")
        except FileNotFoundError:
            print("Warning: Tiny-ImageNet weights not found, starting from scratch or ImageNet pre-training.")
            
    if args.dataset == 'facescrub':
        # Load a pre-trained model on Tiny-ImageNet if available for better starting point
        try:
            base_model.load_state_dict(torch.load('/checkpoints/resnet18_facescrub.pth'))
            print("Loaded pre-trained weights for FaceScrub.")
        except FileNotFoundError:
            print("Warning: FaceScrub weights not found, starting from scratch or ImageNet pre-training.")

    client_model, server_model, defense_module, split_nn = None, None, None, None
    optimizer, scheduler = None, None
    task_loss_fn = nn.CrossEntropyLoss()
    recon_loss_fn = nn.MSELoss()
    best_test_acc = 0.0

    split_logic = args.cutting_layer if 'vgg' in args.arch else args.split_layer
    temp_split = SplitNN(base_model, split_logic)
    dummy_input = torch.randn(2, input_channels, img_size, img_size)
    intermediate_shape = temp_split.client_model(dummy_input).shape
    intermediate_channels = intermediate_shape[1]
    print(f"Detected {intermediate_channels} channels at split point. Smashed data shape: {intermediate_shape}")

    if args.method in ['sae', 'rand_topk']:
        client_model = temp_split.client_model.to(device)
        server_model = temp_split.server_model.to(device)
        
        if args.method == 'sae':
            defense_module = ConvTopKSAE(num_channels=intermediate_channels, hidden_channel_scale=args.hidden_scale, top_p=args.top_p, mask_epsilon=args.mask_epsilon).to(device)
        else: # rand_topk
            defense_module = RandomizedTopKBaseline(top_p=args.top_p * args.hidden_scale, mask_epsilon=args.mask_epsilon).to(device)
        
        full_model_for_attack = nn.Sequential(client_model, SAE_Wrapper(defense_module), server_model).to(device)
        trainable_params = list(client_model.parameters()) + list(defense_module.parameters()) + list(server_model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)



    # ==========================================================================
    # == 2. UNIFIED TRAINING & TESTING LOOP
    # ==========================================================================
    warmup_epochs = args.noise_warm_epochs   # ← modify if you want a different warm‑up length

    for epoch in range(1, args.epochs + 1):
        # ---- warm‑up factor ---------------------------------------------------
        warm_factor = 1.0 if warmup_epochs == 0 else min(1.0, epoch / warmup_epochs)

        # --- Training Phase ----------------------------------------------------
        if args.method in ['sae', 'rand_topk']:
            cal_sigma_list = []
            client_model.train(); server_model.train(); defense_module.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [SAE Training]", ncols=120)
            for imgs, labels in pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()

                client_feats = client_model(imgs)
                reconstruction, sparse_code = defense_module(client_feats)

                if args.noise_aware_training and args.noise_sigma > 0:
                    cal_sigma = calibrate_gaussian_noise_for_value_channel(
                        sparse_code, args.value_channel_budget)
                    cal_sigma *= warm_factor            # ← warm‑up applied here
                    cal_sigma_list.append(cal_sigma)
                    noise = torch.randn_like(sparse_code) * cal_sigma
                    noise_mask = (sparse_code != 0).float()
                    noised_sparse = sparse_code + noise * noise_mask
                    
               
                    
                    final_reconstruction = defense_module.decode(noised_sparse)
                else:
                    final_reconstruction = reconstruction

                server_output = server_model(final_reconstruction)
                loss_task = task_loss_fn(server_output, labels)
                loss_recon = recon_loss_fn(reconstruction, client_feats) if args.method == 'sae' else torch.tensor(0.0).to(device)
                total_loss = args.task_loss_weight * loss_task + args.recon_loss_weight * loss_recon
                total_loss.backward(); optimizer.step()
                pbar.set_postfix({"L_Task": f"{loss_task.item():.4f}", "L_Recon": f"{loss_recon.item():.4f}"})
    

        # -------------------------- Testing Phase ------------------------------
        start_time = time.time()
        correct, total = 0, 0
        if args.method in ['sae', 'rand_topk']:
            client_model.eval(); server_model.eval(); defense_module.eval()
            with torch.no_grad():
                for test_imgs, test_labels in tqdm(test_loader, desc=f"Testing Epoch {epoch}", leave=False, ncols=100):
                    test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)
                    client_feats = client_model(test_imgs)
                    reconstruction, sparse_code = defense_module(client_feats)
                    if args.noise_aware_training:
                        cal_sigma = calibrate_gaussian_noise_for_value_channel(
                            sparse_code, args.value_channel_budget) * warm_factor  # ← warm‑up applied here (same coeff)
                        noise = torch.randn_like(sparse_code) * cal_sigma
                        noise_mask = (sparse_code != 0).float()
                        noised_sparse = sparse_code + noise * noise_mask
                        if args.use_nnc and epoch >= args.epochs:
                            bitstream = encode(noised_sparse, args=nnc_config, quantize_only=nnc_config["quantize_only"])
                            noised_sparse = torch.tensor(decode(bitstream, tensor_id=nnc_config["tensor_id"]), device=device)
                        #####################################
                        end_time = time.time()
                        elapsed_time = end_time - start_time#
                        #####################################
                        reconstruction = defense_module.decode(noised_sparse)
                    outputs = server_model(reconstruction)
                    pred = outputs.argmax(dim=1); total += test_labels.size(0)
                    correct += (pred == test_labels).sum().item()

        test_acc = 100 * correct / total
        

        print(f"[Timing] Inference time for epoch {epoch}: {elapsed_time:.2f} seconds")
        print(f"Epoch {epoch}/{args.epochs} | Test Acc: {test_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        

        if test_acc > best_test_acc:
            print(f"  ✨ New best model! Prev: {best_test_acc:.2f}%. Saving checkpoints…")
            best_test_acc = test_acc
            if args.method in ['sae', 'rand_topk']:
                torch.save(client_model.state_dict(), f'{args.client_ckpt}_{args.mask_epsilon}_{args.value_channel_budget}_{args.top_p}.pt')
                torch.save(server_model.state_dict(), f'{args.server_ckpt}_{args.mask_epsilon}_{args.value_channel_budget}_{args.top_p}.pt')
                torch.save(defense_module.state_dict(), f'{args.sae_ckpt}_{args.mask_epsilon}_{args.value_channel_budget}_{args.top_p}.pt' )
            else:
                torch.save(split_nn.client_model.state_dict(), args.client_ckpt)
                torch.save(split_nn.server_model.state_dict(), args.server_ckpt)
        if scheduler: scheduler.step()

    print(f"\n🏆 Training finished. Best test accuracy: {best_test_acc:.2f}%")

    # ==========================================================================
    # == 3. GAN INVERSION ATTACK (CONDITIONAL)
    # ==========================================================================
    if args.run_attack:
        print("\n--- Initializing GAN Inversion Attack ---")
        #attack_client, attack_defense = split_nn.client_model, defense_module
        
        if args.method in ['sae', 'rand_topk']:
            # Models are already separate, just load the best ones
            attack_client = client_model
            attack_defense = defense_module

        run_gan_inversion_attack(args, attack_client, attack_defense if attack_defense else None, 
                                 train_loader, test_loader, device)

    # --- Final Metrics Reporting ---
    global stop_monitoring
    stop_monitoring = True
    time.sleep(1)
    
    print("\n--- Execution Metrics ---")
    end_time = time.monotonic()
    total_time_seconds = end_time - start_time
    print(f"Total Execution Time: {time.strftime('%H:%M:%S', time.gmtime(total_time_seconds))}")
    
    peak_cpu_memory_mb = peak_cpu_memory / (1024 * 1024)
    print(f"Peak CPU Memory (RAM) Usage: {peak_cpu_memory_mb:.2f} MB")

    if use_cuda:
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"Peak GPU Memory Usage: {peak_memory_mb:.2f} MB")
    
    print("-------------------------\n")


# ==============================================================================
# == GAN ATTACK HELPER FUNCTIONS
# ==============================================================================
def add_spectral_norm(model):
    for name, layer in model.named_children():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            setattr(model, name, torch.nn.utils.spectral_norm(layer))
        else:
            add_spectral_norm(layer)

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

def gradient_penalty(x, y, DG, device):
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape, device=device)
    z = x + alpha * (y - x)
    z.requires_grad = True
    o = DG(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size(), device=device), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()
    return gp

def get_PSNR(real, fake):
    mse = torch.mean((real - fake) ** 2)
    if mse == 0: return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def denormalize(x, dataset):
    if dataset == "tiny_imagenet":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif dataset == "facescrub":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset == "cifar10":
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5] # common CIFAR10 normalization
    else: return torch.clamp(x, 0, 1)

    tensor = x.clone().permute(0, 2, 3, 1) # B, H, W, C
    tensor = tensor * torch.tensor(std, device=x.device) + torch.tensor(mean, device=x.device)
    return torch.clamp(tensor.permute(0, 3, 1, 2), 0, 1) # B, C, H, W


# ==============================================================================
# == GAN INVERSION ATTACK MAIN FUNCTION
# ==============================================================================
def run_gan_inversion_attack(args, client_model, defense_module, train_loader, val_loader, device):
    """
    Main function to run the GAN-based model inversion attack.
    """
    # --- Setup output directories ---
    exp_name = f"gan_attack_{args.method}_mse{args.mse_weight}_noise{args.gan_noise_strength}_lr{args.gan_lr}"
    save_dir_base = os.path.join(args.gan_output_dir, args.dataset, args.arch, args.method)
    save_model_dir = os.path.join(save_dir_base, 'models', exp_name)
    save_img_dir = os.path.join(save_dir_base, 'images', exp_name)
    log_file_path = os.path.join(save_dir_base, f'{exp_name}_log.txt')
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)
    with open(log_file_path, 'w') as f:
        f.write("GAN Inversion Attack Log\n" + "="*25 + "\n")
        for key, val in vars(args).items(): f.write(f"{key}: {val}\n")
        f.write("="*25 + "\n")

    print(f"--- Training GAN Inversion for defense '{args.method}' ---")
    print(f"--- Results will be saved in {save_dir_base}/{exp_name} ---")

    # --- Freeze the defense models ---
    client_model.eval().to(device); freeze(client_model)
    if defense_module is not None: defense_module.eval().to(device); freeze(defense_module)
    print("Loaded and froze client and defense models for attack.")
    num_classes = len(train_loader.dataset.classes)
    # --- Determine feature size dynamically for the GAN Generator ---
    with torch.no_grad():
        dummy_img, _ = next(iter(train_loader))
        dummy_img = dummy_img.to(device)
        client_feats = client_model(dummy_img)

        # Get the actual intermediate representation `z` that the attacker sees
        if isinstance(defense_module, (ConvTopKSAE, RandomizedTopKBaseline)):
            _, z = defense_module(client_feats)
        elif defense_module is not None:
            z = defense_module(client_feats)
        else: # baseline
            z = client_feats

        feature_size = z.shape # (B, C, H, W)
        image_size = dummy_img.shape # (B, C, H, W)
    
    print(f"Detected activation feature shape (for GAN input): {feature_size}")
    print(f"Original image shape (for GAN output): {image_size}")

    # --- Initialize GAN models ---
    print("--- Initializing GAN models ---")
    G = architectures.res_normN_AE(
        N=8, internal_nc=64,
        input_nc=feature_size[1], output_nc=image_size[1],
        input_dim=feature_size[2], output_dim=image_size[2],
        activation="sigmoid"
    ).to(device)
    DG = DGWGAN32().to(device) # Discriminator for 32x32 or 64x64 images
    
    # --- Setup Optimizers and Schedulers ---
    dg_optimizer = optim.Adam(DG.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(G.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    dg_scheduler = optim.lr_scheduler.CosineAnnealingLR(dg_optimizer, T_max=args.gan_epochs)
    g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=args.gan_epochs)

    # --- Main Training Loop ---
    ssim_loss_fn = SSIM()
    mse_loss_fn = nn.MSELoss().to(device)
    step = 0
    
    print("\n--- Starting GAN Training ---")
    for epoch in range(args.gan_epochs):
        start_time = time.time()
        g_losses, d_losses, mse_losses, ssim_losses, psnr_losses = [], [], [], [], []
        
        pbar = tqdm(train_loader, desc=f"GAN Epoch {epoch+1}/{args.gan_epochs}", ncols=120)
        for i, (imgs, _) in enumerate(pbar):
            step += 1
            imgs = imgs.to(device)

            # --- Train Discriminator ---
            freeze(G); unfreeze(DG)
            DG.train()
            # Generate the activation that the GAN will invert
            with torch.no_grad():
                client_feats = client_model(imgs)
                if isinstance(defense_module, (ConvTopKSAE, RandomizedTopKBaseline)):
                    _, z_clean = defense_module(client_feats)
                elif defense_module is not None:
                    z_clean = defense_module(client_feats)
                else: z_clean = client_feats
                
                if args.noise_aware_training:
                    noise_scale = calibrate_gaussian_noise_for_value_channel(z_clean, args.value_channel_budget)
                    noise = noise_scale * torch.randn_like(z_clean)
                    if isinstance(defense_module, (ConvTopKSAE, RandomizedTopKBaseline)):
                        noise_mask = (z_clean != 0).float()
                        z = z_clean + (noise_mask*noise)
                    else:
                        
                        z = z_clean + noise
                else:
                    z = z_clean

            f_imgs = G(z).detach()
            de_imgs = denormalize(imgs, args.dataset)

            r_logit = DG(de_imgs) 
            f_logit = DG(f_imgs)
            gp = gradient_penalty(de_imgs.data, f_imgs.data, DG, device)
            wd = r_logit.mean() - f_logit.mean()
            dg_loss = -wd + gp * 10.0

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()
            d_losses.append(dg_loss.item())

            # --- Train Generator ---
            if step % args.gan_critic_steps == 0:
                freeze(DG); unfreeze(G)
                G.train() 
                f_imgs_g = G(z)
                logit_dg = DG(f_imgs_g)

                adv_loss = -logit_dg.mean()
                recon_loss = mse_loss_fn(f_imgs_g, de_imgs)
                g_loss = adv_loss + args.mse_weight * recon_loss

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                g_losses.append(g_loss.item())

                with torch.no_grad():
                    mse_losses.append(recon_loss.item())
                    ssim_losses.append(1 - ssim_loss_fn(f_imgs_g, de_imgs).item()) # ssim lib returns 1-ssim
                    psnr_losses.append(get_PSNR(f_imgs_g, de_imgs).item())
            
            pbar.set_postfix({
                "D_Loss": f"{np.mean(d_losses):.3f}", 
                "G_Loss": f"{np.mean(g_losses):.3f}",
                "MSE": f"{np.mean(mse_losses):.4f}"
            })

        # --- End of Epoch ---
        dg_scheduler.step(); g_scheduler.step()
        
        avg_g_loss, avg_d_loss = np.mean(g_losses), np.mean(d_losses)
        avg_mse, avg_ssim, avg_psnr = np.mean(mse_losses), np.mean(ssim_losses), np.mean(psnr_losses)
        
        log_msg_train = (f"Epoch:{epoch+1:03d} | D_loss:{avg_d_loss:.4f} | G_loss:{avg_g_loss:.4f} | "
                         f"Train_MSE:{avg_mse:.4f} | Train_SSIM:{avg_ssim:.4f} | Train_PSNR:{avg_psnr:.2f}")
        print(log_msg_train)
        with open(log_file_path, 'a') as f: f.write(log_msg_train + '\n')

        # --- Validation and Saving Images ---
        if (epoch + 1) % 10 == 0:
            G.eval()
            with torch.no_grad():
                val_imgs, _ = next(iter(val_loader))
                val_imgs = val_imgs.to(device)
                
                client_feats_val = client_model(val_imgs)
                if isinstance(defense_module, (ConvTopKSAE, RandomizedTopKBaseline)):
                    _, z_val_clean = defense_module(client_feats_val)
                elif defense_module is not None:
                    z_val_clean = defense_module(client_feats_val)
                else: z_val_clean = client_feats_val
                
                if args.noise_aware_training:
                    noise_scale = calibrate_gaussian_noise_for_value_channel(z_val_clean, args.value_channel_budget)
                    noise = noise_scale * torch.randn_like(z_val_clean)
                    if isinstance(defense_module, (ConvTopKSAE, RandomizedTopKBaseline)):
                        noise_mask = (z_val_clean != 0).float()
                        z_val = z_val_clean + (noise_mask*noise)
                    else:
                        z_val = z_val_clean + noise
                else:
                    z_val = z_val_clean
                
                fake_val_imgs = G(z_val)
                real_val_imgs_denorm = denormalize(val_imgs, args.dataset)

                save_tensor_images(torch.cat([
                    real_val_imgs_denorm.detach()[:16], 
                    fake_val_imgs.detach()[:16]
                ], dim=0), os.path.join(save_img_dir, f"epoch_{epoch+1}.png"), nrow=16)

                val_mse = mse_loss_fn(fake_val_imgs, real_val_imgs_denorm).item()
                val_ssim = 1 - ssim_loss_fn(fake_val_imgs, real_val_imgs_denorm)
                val_psnr = get_PSNR(fake_val_imgs, real_val_imgs_denorm).item()
                
                log_msg_val = f"  => VAL | MSE:{val_mse:.4f} | SSIM:{val_ssim:.4f} | PSNR:{val_psnr:.2f}"
                print(log_msg_val)
                with open(log_file_path, 'a') as f: f.write(log_msg_val + '\n')
            
            print('  => Saving GAN model checkpoints...')
            torch.save(G.state_dict(), os.path.join(save_model_dir, f"G_epoch_{epoch+1}.pth"))
            torch.save(DG.state_dict(), os.path.join(save_model_dir, f"D_epoch_{epoch+1}.pth"))
    
   

    # Load re-identification model
    if args.dataset == 'facescrub':
        base_model = models.resnet18()
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Linear(num_ftrs, num_classes)
        base_model.load_state_dict(torch.load('/checkpoints/resnet18_facescrub.pth'))
        base_model = base_model.to(device)
        base_model.eval()

        correct, total = 0, 0

        with torch.no_grad():
            for val_imgs, val_labels in tqdm(val_loader, desc="Re-identification Eval", ncols=100):
                val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)

                # Generate intermediate features
                client_feats_val = client_model(val_imgs)
                if isinstance(defense_module, (ConvTopKSAE, RandomizedTopKBaseline)):
                    _, z_val_clean = defense_module(client_feats_val)
                elif defense_module is not None:
                    z_val_clean = defense_module(client_feats_val)
                else:
                    z_val_clean = client_feats_val

                # Apply noise if enabled
                if args.noise_aware_training:
                    noise_scale = calibrate_gaussian_noise_for_value_channel(z_val_clean, args.value_channel_budget)
                    noise = noise_scale * torch.randn_like(z_val_clean)
                    if isinstance(defense_module, (ConvTopKSAE, RandomizedTopKBaseline)):
                        noise_mask = (z_val_clean != 0).float()
                        z_val = z_val_clean + (noise_mask * noise)
                    else:
                        z_val = z_val_clean + noise
                else:
                    z_val = z_val_clean

                # Generate reconstructed images
                fake_val_imgs = G(z_val)

                # Pass reconstructed images through re-identification model
                outputs = base_model(fake_val_imgs)
                preds = outputs.argmax(dim=1)

                # Update correct and total
                correct += (preds == val_labels).sum().item()
                total += val_labels.size(0)

        # Compute and report re-identification accuracy
        reid_acc = 100 * correct / total
        print(f"[ReID Eval] Re-identification Accuracy on Reconstructions: {reid_acc:.2f}%")

    
    print("--- GAN Attack Training complete. Final models saved. ---")


if __name__ == '__main__':
    main()