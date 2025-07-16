# Privacy CI

Official code for our paper: *Leveraging Sparsity for Privacy in Collaborative Inference* (WACV 2025).

## Setup

Install dependencies:

```bash
pip install nncodec torch torchvision
```

## Run Experiments

To run all experiments with default settings:

```bash
bash run.sh
```

## Datasets

Please download and place the datasets manually:

- **FaceScrub**: [https://www.kaggle.com/datasets/rajnishe/facescrub-full](https://www.kaggle.com/datasets/rajnishe/facescrub-full)
- **Tiny-ImageNet**: [https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)

Place the extracted folders in the appropriate data path used by the scripts (`run.sh`).
