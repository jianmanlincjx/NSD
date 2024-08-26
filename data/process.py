import os
import cv2

def is_image_broken(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return True
        print(f"{image_path} shape: {img.shape}")
        return False
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return True

def is_text_broken(text_path):
    try:
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
        return not content  # Return True if content is empty
    except Exception as e:
        print(f"Error reading {text_path}: {e}")
        return True

def is_mask_broken(mask_path_list):
    for mask_path in mask_path_list:
        if not os.path.exists(mask_path):
            return True
        try:
            mask = cv2.imread(mask_path)
            if mask is None:
                return True
            print(f"{mask_path} shape: {mask.shape}")
        except Exception as e:
            print(f"Error reading {mask_path}: {e}")
            return True
    return False

def delete_broken_paths(image_root, text_root, mask_root, img_list, vid_list):
    missing_or_broken_count = 0

    for img in img_list:
        image_path = os.path.join(image_root, img)
        text_path = image_path.replace('image', 'text').replace('png', 'txt')
        mask_path_list = [os.path.join(mask_root, i, img) for i in vid_list]
        
        if (not os.path.exists(image_path) or is_image_broken(image_path) or
            not os.path.exists(text_path) or is_text_broken(text_path) or
            is_mask_broken(mask_path_list)):
            missing_or_broken_count += 1
            # Delete the image, text, and mask files if they exist
            if os.path.exists(image_path):
                os.remove(image_path)
            if os.path.exists(text_path):
                os.remove(text_path)
            for mask_path in mask_path_list:
                if os.path.exists(mask_path):
                    os.remove(mask_path)
    
    return missing_or_broken_count

if __name__ == "__main__":
    image_root = '/data0/JM/code/BrushNet/data/data_train_big/image'
    text_root = '/data0/JM/code/BrushNet/data/data_train_big/text'
    mask_root = '/data0/JM/code/BrushNet/data/data_train_big/mask'
    
    img_list = sorted(os.listdir(image_root))
    vid_list = os.listdir(mask_root)
    
    missing_or_broken_count = delete_broken_paths(image_root, text_root, mask_root, img_list, vid_list)
    
    print(f"Number of instances where paths were missing or files were broken and were deleted: {missing_or_broken_count}")
