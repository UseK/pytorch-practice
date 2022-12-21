from glob import glob
from PIL import Image

images = glob("data/mydata/images/*")
for f in images:
    f_name = f.split("/")[-1]
    resized_path = f"data/mydata/resized_512_512/{f_name}"
    print(resized_path)
    Image.open(f).resize((512, 512)).save(resized_path)