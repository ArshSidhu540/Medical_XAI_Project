import os
import torch
import torch.nn as nn
from src.models.model import get_brain_tumor_model

def main():
    print("⚡ Starting Post-Training Static Quantization Pipeline...")
    
    # 1. Initialize standard model architecture
    model_fp32 = get_brain_tumor_model(num_classes=4, pretrained=False)
    
    weights_path = 'models/best_brain_tumor_model.pth'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Missing source baseline weights at '{weights_path}'. Run train.py first!")
        
    model_fp32.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model_fp32.eval()
    
    # 2. Configure Quantization Modules
    # Attach backend configuration for desktop/server CPU optimization (x86 architectures)
    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    
    # Prepare the network by inserting structural observers to track activation distributions
    model_prepared = torch.ao.quantization.prepare(model_fp32, inplace=False)
    
    # 3. Calibrate the Observers
    # Simulate passing 5 fake MRI image tensors through the network to calibrate quantization boundaries
    print("📈 Calibrating activation scopes using synthetic calibration tensors...")
    with torch.no_grad():
        for _ in range(5):
            dummy_mri = torch.randn(1, 3, 224, 224)
            model_prepared(dummy_mri)
            
    # 4. Convert to Quantized Representation (float32 -> int8)
    print("🗜️ Compressing weight parameters into 8-bit integer maps...")
    model_int8 = torch.ao.quantization.convert(model_prepared, inplace=False)
    
    # 5. Serialize lightweight model
    output_path = 'models/quantized_brain_tumor_model.pth'
    torch.save(model_int8.state_dict(), output_path)
    
    # 6. Evaluate Structural Footprints
    size_fp32 = os.path.getsize(weights_path) / (1024 * 1024)
    size_int8 = os.path.getsize(output_path) / (1024 * 1024)
    compression_ratio = (1 - (size_int8 / size_fp32)) * 100
    
    print("\n--- 📊 Quantization Footprint Report ---")
    print(f"📦 Original FP32 Model Size:   {size_fp32:.2f} MB")
    print(f"📦 Compressed INT8 Model Size: {size_int8:.2f} MB")
    print(f"📉 Total Memory Footprint Slashed by: {compression_ratio:.2f}%")
    print(f"💾 Quantized asset compiled successfully at: {output_path}")

if __name__ == '__main__':
    main()