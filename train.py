import os 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.utils.dataset import BrainTumorDataset
from src.models.model import get_brain_tumor_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Runs through the entire training dataset once (one epoch)."""
    model.train() # Set model to training mode (enables dropout/batchnorm updates)
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for images, labels in dataloader:
        # Move our tensor data to GPU if available, otherwise stay on CPU
        images, labels = images.to(device), labels.to(device)
        
        # 1. Clear previous gradients so they don't accumulate incorrectly
        optimizer.zero_grad()
        
        # 2. Forward pass: Feed the images into the model to get raw prediction scores
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 3. Backward pass: Calculate the gradients (amplify what worked, diminish what didn't)
        loss.backward()
        
        # 4. Update step: Adjust the model weights slightly
        optimizer.step()
        
        # Track training statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1) # Find the class index with the highest score
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def validate_model(model, dataloader, criterion, device):
    """Runs through the validation/testing dataset to evaluate true performance."""
    model.eval() # Set model to evaluation mode (freezes dropout/batchnorm)
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Disable gradient tracking entirely to save memory and make execution faster
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
    val_loss = running_loss / total_samples
    val_acc = correct_predictions / total_samples
    return val_loss, val_acc

def main():
    # Setup hardware acceleration (Use NVIDIA CUDA if present, else fallback to standard CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Training Engine initialization. Using device target: {device}")
    
    # 1. Define standard pipelines
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Instantiate datasets and loaders
    train_dataset = BrainTumorDataset(root_dir='data/raw/Training', transform=train_transform)
    test_dataset = BrainTumorDataset(root_dir='data/raw/Testing', transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Load our customized model architecture onto the designated device
    model = get_brain_tumor_model(num_classes=4, pretrained=True).to(device)
    
    # 4. Define Loss configuration & Optimization Hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Low learning rate for fine-tuning
    
    # 5. The Training Loop (Let's run 5 initial epochs to watch it learn)
    epochs = 5
    best_val_acc = 0.0
    
    print("\n🚀 Starting Training Pipeline execution...\n")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, test_loader, criterion, device)
        
        print(f"📊 Epoch [{epoch+1}/{epochs}]")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        
        # Save the absolute best weights configuration based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Ensure folder structure exists
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_brain_tumor_model.pth')
            print("   💾 New optimal performance achieved. Weights checkpoint saved!")
            
    print(f"\n🎉 Training complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")

if __name__ == '__main__':
    main()