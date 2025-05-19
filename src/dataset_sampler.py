import os
import shutil
import random
import yaml
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_file_list(directory):
    """Get list of files in directory (images or labels)"""
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return []
    
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def sample_dataset(src_folder, total_count, ratios, output_dir):
    """
    Sample dataset according to specified ratios
    
    Args:
        src_folder: Source data folder
        total_count: Total number of images to sample
        ratios: List of ratios for train:test:valid
        output_dir: Directory to save sampled dataset
    """
    # Calculate counts for each split based on ratios
    total_ratio = sum(ratios)
    train_count = int(total_count * ratios[0] / total_ratio)
    test_count = int(total_count * ratios[1] / total_ratio)
    valid_count = total_count - train_count - test_count
    
    logger.info(f"Sampling dataset: {train_count} training, {test_count} test, {valid_count} validation images")
    
    # Directly construct paths from the source folder
    src_train_img = os.path.join(src_folder, 'train', 'images')
    src_test_img = os.path.join(src_folder, 'test', 'images')
    src_valid_img = os.path.join(src_folder, 'valid', 'images')
    
    src_train_labels = os.path.join(src_folder, 'train', 'labels')
    src_test_labels = os.path.join(src_folder, 'test', 'labels')
    src_valid_labels = os.path.join(src_folder, 'valid', 'labels')

    logger.info(f"Source directories:")
    logger.info(f"  Train images: {src_train_img}")
    logger.info(f"  Train labels: {src_train_labels}")
    logger.info(f"  Test images: {src_test_img}")
    logger.info(f"  Test labels: {src_test_labels}")
    logger.info(f"  Valid images: {src_valid_img}")
    logger.info(f"  Valid labels: {src_valid_labels}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    dst_train_img = os.path.join(output_dir, 'train', 'images')
    dst_test_img = os.path.join(output_dir, 'test', 'images')
    dst_valid_img = os.path.join(output_dir, 'valid', 'images')
    dst_train_labels = os.path.join(output_dir, 'train', 'labels')
    dst_test_labels = os.path.join(output_dir, 'test', 'labels')
    dst_valid_labels = os.path.join(output_dir, 'valid', 'labels')
    
    for directory in [dst_train_img, dst_test_img, dst_valid_img, 
                     dst_train_labels, dst_test_labels, dst_valid_labels]:
        os.makedirs(directory, exist_ok=True)
    
    # Get list of available files (from labels)
    train_labels = get_file_list(src_train_labels)
    test_labels = get_file_list(src_test_labels)
    valid_labels = get_file_list(src_valid_labels)
    
    logger.info(f"Found {len(train_labels)} training, {len(test_labels)} test, {len(valid_labels)} validation label files")
    
    # Sample files
    sampled_train = random.sample(train_labels, min(train_count, len(train_labels))) if train_labels else []
    sampled_test = random.sample(test_labels, min(test_count, len(test_labels))) if test_labels else []
    sampled_valid = random.sample(valid_labels, min(valid_count, len(valid_labels))) if valid_labels else []
    
    # Function to copy files (both image and label)
    def copy_files(file_list, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
        copied_count = 0
        missing_images = 0
        for label_file in file_list:
            # Get corresponding image filename
            # Assuming image has same base name but different extension
            base_name = os.path.splitext(label_file)[0]
            
            # Try common image extensions
            found_image = False
            for ext in ['.jpg', '.jpeg', '.png']:
                img_file = base_name + ext
                img_path = os.path.join(src_img_dir, img_file)
                
                if os.path.exists(img_path):
                    # Copy image
                    shutil.copy2(img_path, os.path.join(dst_img_dir, img_file))
                    # Copy label
                    shutil.copy2(os.path.join(src_label_dir, label_file), 
                                os.path.join(dst_label_dir, label_file))
                    copied_count += 1
                    found_image = True
                    break
            
            if not found_image:
                missing_images += 1
        
        if missing_images > 0:
            logger.warning(f"Could not find images for {missing_images} label files")
            
        return copied_count
    
    # Copy files for each split
    train_copied = copy_files(sampled_train, src_train_img, src_train_labels, 
                             dst_train_img, dst_train_labels)
    test_copied = copy_files(sampled_test, src_test_img, src_test_labels, 
                            dst_test_img, dst_test_labels)
    valid_copied = copy_files(sampled_valid, src_valid_img, src_valid_labels, 
                             dst_valid_img, dst_valid_labels)
    
    logger.info(f"Copied {train_copied} training, {test_copied} test, {valid_copied} validation files")
    
    # Create a new data.yaml file
    # Let's read the original data.yaml to keep the class names and other info
    with open(os.path.join(src_folder, 'data.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        
    # Update paths
    config['train'] = '../train/images'
    config['val'] = '../valid/images'
    config['test'] = '../test/images'
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created new data.yaml file at {os.path.join(output_dir, 'data.yaml')}")

def main():
    parser = argparse.ArgumentParser(description="Sample YOLO dataset with specific ratios")
    parser.add_argument("-s", "--src", type=str, required=True, help="Source data folder")
    parser.add_argument("-t", "--total", type=int, required=True, help="Total number of images to sample")
    parser.add_argument("-tr", "--train-ratio", type=int, default=7, help="Training set ratio")
    parser.add_argument("-te", "--test-ratio", type=int, default=1, help="Test set ratio")
    parser.add_argument("-v", "--valid-ratio", type=int, default=2, help="Validation set ratio")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Sample dataset
    sample_dataset(args.src, args.total, 
                  [args.train_ratio, args.test_ratio, args.valid_ratio], args.output)
    
    logger.info("Dataset sampling completed!")

if __name__ == "__main__":
    main()