import sys
import os

# Ensure the root folder is fully recognized by Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.model import get_brain_tumor_model

if __name__ == "__main__":
    print("🚀 Attempting to load the architecture...")
    model = get_brain_tumor_model(num_classes=4, pretrained=True)
    print("🎉 Success! The architecture imported and compiled flawlessly.")