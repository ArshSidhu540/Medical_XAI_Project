import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the data directory (e.g., 'data/Training')
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Define our 4 distinct medical classes
        self.classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        
        # Create a mapping from class name string to an integer index (0, 1, 2, 3)
        # Deep learning models understand numbers, not strings!
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Temporarily check what files are inside the Training/glioma folder:
        test_folder_path = os.path.join(self.root_dir, 'glioma')
        if os.path.exists(test_folder_path):
            print(f"📂 Sample files inside glioma folder: {os.listdir(test_folder_path)[:5]}")
        
        # Gather all image file paths and their matching numerical labels
        self.image_samples = []
        
        for cls_name in self.classes:
            class_folder = os.path.join(self.root_dir, cls_name)
            
            if os.path.exists(class_folder):
                files = os.listdir(class_folder)
                # 🔎 Diagnostic print: Let's see the first 3 files found in this folder
                if len(files) > 0:
                    print(f"📁 Inside '{cls_name}', Python sees these sample files: {files[:3]}")
                else:
                    print(f"⚠️ Warning: The folder '{cls_name}' is completely empty!")

                for filename in files:
                    # Temporarily allow ALL files so we can catch whatever format they are in
                    img_path = os.path.join(class_folder, filename)
                    self.image_samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        """Returns the total number of images found in this dataset split."""
        return len(self.image_samples)

    def __getitem__(self, idx):
        """Fetches a single image and its label using its index."""
        img_path, label = self.image_samples[idx]
        
        # Open the image and convert it to RGB mode
        # Medical MRIs can sometimes save as grayscale; converting ensures a consistent 3-channel input
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations (resizing, converting to tensor, etc.) if provided
        if self.transform:
            image = self.transform(image)
            
        return image, label