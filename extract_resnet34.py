#!/usr/bin/env python3
"""
Extract ResNet34 pretrained weights from timm and save in the format expected by SeerDrive.
This allows you to use standard ImageNet-pretrained ResNet34 weights.
"""

import torch
import timm
from pathlib import Path

def extract_resnet34_weights(output_path: str):
    """
    Extract ResNet34 pretrained weights from timm and save them.
    
    Args:
        output_path: Path where to save the checkpoint
    """
    print("Creating ResNet34 model with pretrained ImageNet weights...")
    
    # Create model with pretrained weights
    model = timm.create_model("resnet34", pretrained=True, features_only=True)
    
    # Get the state dict
    state_dict = model.state_dict()
    
    print(f"Extracted {len(state_dict)} parameters")
    print(f"First few keys: {list(state_dict.keys())[:5]}")
    
    # Save the state dict
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(state_dict, output_path)
    print(f"\n✅ Saved ResNet34 weights to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    return output_path

if __name__ == "__main__":
    import sys
    
    # Default output path
    output_path = "/home/harsh/SeerDrive/ckpts/resnet34.pth"
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    
    print("=" * 60)
    print("Extracting ResNet34 Pretrained Weights")
    print("=" * 60)
    print(f"\nThis will download ImageNet-pretrained ResNet34 from timm")
    print(f"and save it in the format expected by SeerDrive.\n")
    
    try:
        extract_resnet34_weights(output_path)
        print("\n✅ Success! You can now use this checkpoint with SeerDrive.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

