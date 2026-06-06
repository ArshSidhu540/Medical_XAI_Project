import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

def get_brain_tumor_model(num_classes=4, pretrained=True):
    """
    Loads a ResNet-18 architecture and modifies its final layer 
    to output probabilities for our specific brain tumor classes.
    """
    if pretrained:
        # Fetch the best available pre-trained ImageNet weights
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        print("🔄 Loaded ResNet-18 with pre-trained ImageNet weights (Transfer Learning enabled).")
    else:
        model = models.resnet18(weights=None)
        print("🔄 Loaded a blank ResNet-18 (Training from scratch).")
        
    # Find the number of input features going into ResNet's original final layer ('fc')
    in_features = model.fc.in_features
    
    # Replace the final fully connected layer with a fresh linear layer
    # This maps the internal deep features to our 4 target classes
    model.fc = nn.Linear(in_features, num_classes)
    print(f"🎯 Modified model head: Final layer now outputs {num_classes} classes.")
    
    return model

if __name__ == "__main__":
    # Quick structural sanity check to make sure the network compiles locally
    print("Running architectural sanity check...")
    test_model = get_brain_tumor_model(num_classes=4, pretrained=False)
    
    # Simulate 1 fake image batch: [Batch_size=1, Channels=3, Height=224, Width=224]
    dummy_tensor = torch.randn(1, 3, 224, 224)
    output = test_model(dummy_tensor)
    
    print(f"🎬 Structural Check Passed! Output Shape: {output.shape} -> (Expected: [1, 4])")