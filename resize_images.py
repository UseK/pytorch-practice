from glob import glob
from pathlib import Path
from PIL import Image
import os

def process_each_images(input_dir: Path, resized_dirname: str, f):
    images_paths = glob(f"{input_dir}/*")
    resized_dir = input_dir.parent.joinpath(f"{input_dir.name}_{resized_dirname}")
    print(resized_dir)
    for image_path in images_paths:
        img = Image.open(image_path)
        processed = f(img)
        # Image.open(image_path).resize((512, 512)).save(resized_path)
        f_name = image_path.split("/")[-1]
        os.makedirs(resized_dir, exist_ok=True)
        resized_path = resized_dir.joinpath(f_name)
        print(resized_path)
        processed.save(resized_path)

if __name__ == '__main__':
    # process_each_images(Path("data/mydata/images"), "resized_512_512", lambda img: img.resize((512, 512)))
    process_each_images(Path("data/movies/2022121721104602"), "resized_512_512", lambda img: img.resize((512, 512)))