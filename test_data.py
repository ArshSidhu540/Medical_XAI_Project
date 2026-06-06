import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.utils.dataset import BrainTumorDataset

def main():
    # 1. Define Standard Medical Transforms
    # Training needs augmentation to prevent overfitting; Testing only needs standardization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),      # Flip images horizontally with 50% chance
        transforms.RandomRotation(degrees=15),         # Subtle rotation (common in slight head tilts)
        transforms.ToTensor(),                         # Convert PIL Image to PyTorch Tensor (0 to 1)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]) # ImageNet standardization values
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Instantiate our custom dataset splits
    # (Using relative paths pointing to your local data folder)
    print("🔄 Loading datasets...")
    train_dataset = BrainTumorDataset(root_dir='data/raw/Training', transform=train_transform)
    test_dataset = BrainTumorDataset(root_dir='data/raw/Testing', transform=test_transform)

    print(f"✅ Training samples found: {len(train_dataset)}")
    print(f"✅ Testing samples found: {len(test_dataset)}")

    # 3. Create the PyTorch DataLoaders
    # shuffle=True for training ensures the model doesn't learn patterns from the order of images
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 4. Fetch exactly one batch to inspect its structure
    images, labels = next(iter(train_loader))
    print("\n--- DataLoader Inspection ---")
    print(f"Images batch shape: {images.shape} -> [Batch Size, Channels, Height, Width]")
    print(f"Labels batch shape: {labels.shape} -> Contains 32 numerical class indexes")
    print("Everything is structured perfectly! Ready for the model phase.")

if __name__ == '__main__':
    main()