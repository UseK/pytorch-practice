from pathlib import Path
import subprocess
from sys import argv
from os import makedirs


def main(path: Path):
    """
    like $ ffmpeg -i xxx.mp4 xxx/%06d.jpg
    """
    subprocess.run(["ffmpeg", "-version"])
    print(path.stem)
    print(path.parent)
    image_dir = path.parent.joinpath(path.stem)
    makedirs(image_dir, exist_ok=True)
    print(image_dir)
    # subprocess.run("ls", shell=True)
    subprocess.run(["ffmpeg", "-i", f"{path}", f"{image_dir}/%06d.jpg"])



if len(argv) < 2:
    print("Usage: python3 extract_from_movies MOVIE_PATH")
else:
    main(Path(argv[1]))
