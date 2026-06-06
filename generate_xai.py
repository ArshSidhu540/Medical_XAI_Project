import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from src.models.model import get_brain_tumor_model
from src.explainers.gradcam import GradCAM

def main():
    # 1. Setup Environment & Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    
    # 2. Reconstruct Model and Load Trained Weights Checkpoint
    print("🔄 Initializing model architecture...")
    model = get_brain_tumor_model(num_classes=4, pretrained=False)
    
    weights_path = 'models/best_brain_tumor_model.pth'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Could not find model weights at '{weights_path}'. Please run train.py first!")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("✅ Successfully loaded trained model weights!")

    # 3. Locate an Actual Sample from the Testing Folders
    # We will grab the first image available in the testing glioma folder to test
    sample_class = 'glioma'
    sample_folder = f'data/raw/Testing/{sample_class}'
    
    if not os.path.exists(sample_folder) or len(os.listdir(sample_folder)) == 0:
        raise FileNotFoundError(f"❌ No files found in target sample path: {sample_folder}")
        
    sample_filename = os.listdir(sample_folder)[0]
    img_path = os.path.join(sample_folder, sample_filename)
    print(f"📸 Selected sample scan: {img_path}")

    # 4. Standard Preprocessing Pipeline
    # Keep an un-normalized version for structural display, and a normalized one for the neural network
    original_pil = Image.open(img_path).convert('RGB')
    display_img = np.array(original_pil.resize((224, 224))) # Base shape for OpenCV steps [224, 224, 3]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(original_pil).unsqueeze(0).to(device) # Shape: [1, 3, 224, 224]

    # 5. Initialize XAI Engine Target Layer
    # ResNet-18's final convolutional block is named 'layer4'
    xai_engine = GradCAM(model=model, target_layer=model.layer4)
    
    # 6. Execute Forward & Backward pass to build heatmap
    print("🧠 Generating Class Activation Heatmap...")
    heatmap = xai_engine.generate_heatmap(input_tensor)

    # 7. Post-Processing: Upscale and Apply Jet Palette using OpenCV
    # Resize the 2D heatmap up to 224x224
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    # Convert float matrix (0-1) to 8-bit unsigned integers (0-255) for color mapping
    heatmap_8bit = np.uint8(255 * heatmap_resized)
    # Apply standard thermal spectrum lookup table
    heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    # OpenCV uses BGR by default, convert to standard RGB format
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 8. Blend the original MRI anatomy with the generated heatmap
    # Formula: blended = (alpha * original) + (beta * heatmap)
    alpha = 0.6
    blended_visual = cv2.addWeighted(display_img, alpha, heatmap_colored, 1 - alpha, 0)

    # 9. Concatenate side-by-side: [Original Image | Blended Heatmap Explanation]
    final_output_panel = np.hstack((display_img, blended_visual))

    # 10. Write the resulting analytical proof to disk
    output_dir = 'output_visuals'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'gradcam_result.png')
    
    # Convert array back to PIL Image and save
    final_result_image = Image.fromarray(final_output_panel)
    final_result_image.save(output_path)
    
    # Run a quick model inference to print what the model diagnosed
    with torch.no_grad():
        prediction_logits = model(input_tensor)
        predicted_idx = torch.argmax(prediction_logits, dim=1).item()
        
    print("\n--- Diagnostic & XAI Complete ---")
    print(f"🏷️ True Diagnosis Label:     {sample_class}")
    print(f"🤖 AI Predicted Diagnosis:   {classes[predicted_idx]}")
    print(f"💾 Visual analytical proof saved successfully at: {output_path}")

if __name__ == '__main__':
    main()