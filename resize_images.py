from glob import glob
from PIL import Image
import os

def process_each_images(dirname: str, f):
    images_paths = glob("data/mydata/images/*")
    for image_path in images_paths:
        img = Image.open(image_path)
        processed = f(img)
        # Image.open(image_path).resize((512, 512)).save(resized_path)
        f_name = image_path.split("/")[-1]
        os.makedirs(f"data/mydata/{dirname}", exist_ok=True)
        resized_path = f"data/mydata/{dirname}/{f_name}"
        print(resized_path)
        processed.save(resized_path)

if __file__ == '__main__':
    process_each_images("resized_512_512", lambda img: img.resize((512, 512)))