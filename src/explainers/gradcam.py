import torch
import torch.nn as nn
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained PyTorch model (ResNet-18)
            target_layer: The specific layer we want to extract features from (e.g., model.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        
        # Placeholders to store our extracted internal maps
        self.gradients = None
        self.activations = None
        
        # Hook 1: Extract gradients flowing backward through the target layer
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        # Hook 2: Extract activations flowing forward through the target layer
        def forward_hook(module, input, output):
            self.activations = output
            
        # Register hooks onto our chosen layer
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        """Generates a raw 2D intensity heatmap for a specific class prediction."""
        self.model.eval()
        
        # 1. Forward pass to extract features
        output = self.model(input_tensor)
        
        # If no specific class index is requested, explain the model's top choice
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # 2. Backward pass targeting our specific class score
        self.model.zero_grad()
        class_score = output[0][class_idx]
        class_score.backward()
        
        # 3. Pull our hooked data out of memory
        gradients = self.gradients.detach().cpu().numpy()[0]     # Shape: [Channels, H, W]
        activations = self.activations.detach().cpu().numpy()[0] # Shape: [Channels, H, W]
        
        # 4. Global Average Pooling: Compute importance weight for each channel
        weights = np.mean(gradients, axis=(1, 2)) # Shape: [Channels]
        
        # 5. Weighted combination of activation maps
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32) # Base 2D canvas
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # 6. Pass through ReLU: Keep only features that positively influence the diagnosis
        heatmap = np.maximum(heatmap, 0)
        
        # Min-max normalization to scale values perfectly between 0.0 and 1.0
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)
            
        return heatmap
    


class VanillaSaliency:
    def __init__(self, model):
        """
        Args:
            model: Trained PyTorch model
        """
        self.model = model

    def generate_saliency(self, input_tensor, class_idx=None):
        """Generates a pixel-level attribution saliency map."""
        self.model.eval()
        
        # Enforce that the input tensor tracks gradients directly at the pixel level
        input_tensor.requires_grad_with_cv(True) if hasattr(input_tensor, 'requires_grad_with_cv') else setattr(input_tensor, 'requires_grad', True)
        
        # Forward Pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        class_score = output[0][class_idx]
        
        # Backward Pass to calculate derivatives relative to input pixels
        self.model.zero_grad()
        class_score.backward()
        
        # Extract the absolute gradients from the input tensor
        saliency, _ = torch.max(torch.abs(input_tensor.grad.data), dim=1)
        saliency_map = saliency[0].cpu().numpy()
        
        # Normalize between 0.0 and 1.0
        if np.max(saliency_map) != 0:
            saliency_map = saliency_map / np.max(saliency_map)
            
        return saliency_map