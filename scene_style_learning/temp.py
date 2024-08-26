import os
from PIL import Image

def check_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify if image is corrupt
        return True
    except (IOError, SyntaxError) as e:
        print(f"File {file_path} is damaged or corrupted: {e}")
        return False

def check_images_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(directory, filename)
            check_image(file_path)

if __name__ == "__main__":
    directory = "/data1/JM/code/BrushNet/scene_style_learning/data/test"
    check_images_in_directory(directory)
