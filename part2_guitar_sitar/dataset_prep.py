import os
import zipfile
import glob
import shutil
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(SCRIPT_DIR, '..', 'gvs_dataset.zip')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

def process_part2():
    if not os.path.exists(ZIP_PATH):
        print(f"Error: Dataset zip not found at {ZIP_PATH}.")
        return

    print("Extracting zip file...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(DATA_DIR)
        
    extracted_folder = os.path.join(DATA_DIR, 'gvs_dataset')
    
    # Target folders
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    classes = ['Guitar', 'Sitar']
    
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
        
        # Original paths
        orig_train_cls = os.path.join(extracted_folder, 'Train_Imgs', cls)
        orig_test_cls = os.path.join(extracted_folder, 'Test_Imgs', cls)
        
        # 1. Move Test images
        test_imgs = glob.glob(os.path.join(orig_test_cls, '*.*'))
        for img in test_imgs:
            shutil.copy(img, os.path.join(test_dir, cls, os.path.basename(img)))
            
        # 2. Split Train images into Train / Val (85/15 split like part 1)
        train_imgs_all = glob.glob(os.path.join(orig_train_cls, '*.*'))
        
        # Split
        final_train, final_val = train_test_split(train_imgs_all, test_size=0.15, random_state=42)
        
        for img in final_train:
            shutil.copy(img, os.path.join(train_dir, cls, os.path.basename(img)))
        for img in final_val:
            shutil.copy(img, os.path.join(val_dir, cls, os.path.basename(img)))
            
        print(f"Class {cls}: {len(final_train)} train, {len(final_val)} val, {len(test_imgs)} test")

    # Cleanup the extracted gvs_dataset folder
    shutil.rmtree(extracted_folder)
    print(f"Dataset extracted and split into train/val/test -> {DATA_DIR}")

if __name__ == '__main__':
    process_part2()
