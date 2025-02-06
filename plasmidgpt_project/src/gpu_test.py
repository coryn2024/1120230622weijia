import torch

if __name__ == "__main__":
    print("torch version:", torch.__version__)
    print("cuda version:", torch.version.cuda)
    print("GPU is available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

"""
torch version: 2.4.1+cu118
cuda version: 11.8
GPU is available: True
GPU name: NVIDIA GeForce RTX 4060 Laptop GPU
"""