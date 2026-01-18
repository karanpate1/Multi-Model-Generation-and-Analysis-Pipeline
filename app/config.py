import torch
import os

class Config:
   
    # If a GPU is available, use it. Otherwise, force CPU.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # GPU uses float16 (fast/low memory), CPU must use float32
    DTYPE = torch.float32

    # Model IDs
    SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"
    SAM_MODEL_ID = "sam2.1_b.pt" 
    CLIP_MODEL_ID = "ViT-B/32"

    print(f"CONFIG: Running on {DEVICE} ({DTYPE})")