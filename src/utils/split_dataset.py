import os
import shutil
import random

def split_medical_data(base_dir="data/raw", split_ratio=0.8):
    """
    Shuffles and splits the raw medical images into Training (80%) and Testing (20%) folders,
    then cleans up the loose original folders to prevent duplication.
    """
    # The 4 target medical categories we found in your folder
    categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    
    print("🚀 Starting dataset splitting process...")
    
    for category in categories:
        source_folder = os.path.join(base_dir, category)
        
        # Safety check: If the loose folder doesn't exist, skip it (maybe it's already split!)
        if not os.path.exists(source_folder):
            print(True)
            continue
            
        # Get a list of all images inside this specific tumor folder
        images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
        
        # Shuffle the images randomly so our split doesn't accidentally group similar scans together
        random.seed(42)  # Setting a 'seed' ensures this random shuffle happens exactly the same way every time you run it
        random.shuffle(images)
        
        # Calculate our mathematical split point (e.g., 80% of 1000 images = 800)
        split_point = int(len(images) * split_ratio)
        train_images = images[:split_point]
        test_images = images[split_point:]
        
        print(f"📦 Category '{category}': Found {len(images)} images. Splitting into {len(train_images)} Train and {len(test_images)} Test...")
        
        # Move files to their permanent homes inside Training
        for img in train_images:
            src_path = os.path.join(source_folder, img)
            dest_path = os.path.join(base_dir, 'Training', category, img)
            shutil.move(src_path, dest_path)
            
        # Move files to their permanent homes inside Testing
        for img in test_images:
            src_path = os.path.join(source_folder, img)
            dest_path = os.path.join(base_dir, 'Testing', category, img)
            shutil.move(src_path, dest_path)
            
        # Clean up: Delete the now empty original loose folder
        os.rmdir(source_folder)
        print(f"✅ Successfully processed and cleaned up '{category}' folder.")

    print("\n🎉 Dataset successfully split and deduplicated! You are ready for training preprocessing.")

if __name__ == "__main__":
    split_medical_data()