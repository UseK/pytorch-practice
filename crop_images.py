
from PIL import Image
from resize_images import process_each_images


def crop_image(img: Image):
    return img.crop((600, 30, 670, 70))

process_each_images("crop_600_30_670_70", crop_image)