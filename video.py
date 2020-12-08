import imageio
import numpy as np
import os.path
from PIL import Image
from progress.bar import Bar
import cv2 as cv

# Local Modules
from constants import MAX_COLOR
from dithering import floyd_steinberg_dithering, ordered_dithering
import utils

VIDEOS_DIR = "videos"
MAX_INTENSITY = 255
VIDEO_FILENAME = f"{VIDEOS_DIR}/anim_final.mp4"
OUT_VIDEO_FILENAME = f"{VIDEOS_DIR}/out.mp4"
MAX_QUALITY = 95
# duration of video in seconds
DURATION = 4
FPS = 12
RGB_WEIGHT = np.array([0.2989, 0.5870, 0.1140])


def main():
    timer = utils.Timer()
    timer.start()
    # For creating video
    if not os.path.exists(VIDEOS_DIR):
        os.mkdir(VIDEOS_DIR)
    counter = 0
    total_frames = DURATION * FPS
    bar = Bar("Processing...", max=total_frames, suffix='%(percent)d%%')
    bar.check_tty = False
    # Read frames from video
    reader = imageio.get_reader(VIDEO_FILENAME)
    writer = imageio.get_writer(
       OUT_VIDEO_FILENAME, format="mp4", mode='I', fps=FPS
    )
    for i, frame in enumerate(reader):
        w, h, _ = frame.shape
        frame = np.array(frame) / MAX_COLOR
        grayscale = np.dot(frame, RGB_WEIGHT)
        img_arr = floyd_steinberg_dithering(grayscale)
        img = Image.fromarray(img_arr)
        img = img.convert("RGB")
        # Append rendered image into video
        img_arr = np.array(img)
        writer.append_data(img_arr)
        counter += 1
        bar.next()
    bar.finish()
    print("Writing video")
    writer.close()
    timer.stop()
    print(f"Total time spent {timer}")


if __name__ == '__main__':
    main()
