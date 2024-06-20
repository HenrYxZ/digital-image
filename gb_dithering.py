import imageio
import numpy as np
import os.path
from progress.bar import Bar

# Local Modules
from constants import MAX_COLOR, RGB_CHANNELS
from dithering import floyd_steinberg_dithering
import utils


VIDEOS_DIR = "videos"
MAX_INTENSITY = 255
VIDEO_FILENAME = f"{VIDEOS_DIR}/anim_final_raytraced.mp4"
OUT_VIDEO_FILENAME = f"{VIDEOS_DIR}/out.mp4"
MAX_QUALITY = 95
FPS = 12
SCALE = (0.0, 0.33, 0.66, 1.0)
PALETTE = {
    0: np.array([41, 65, 57], dtype=np.uint8),
    84: np.array([57, 89, 74], dtype=np.uint8),
    168: np.array([90, 121, 66], dtype=np.uint8),
    255: np.array([123, 130, 16], dtype=np.uint8)
}
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 144
RGB_WEIGHT = np.array([0.2989, 0.5870, 0.1140])


def fit_screen(w0: int, h0: int):
    """
    Fit the screen by returning new width and height. Calculate the width if the
    screen height is used, and the height if the screen width is used, and use
    the one that fits.
    """
    w1 = (h0 / w0) * SCREEN_HEIGHT
    h1 = (w0 / h0) * SCREEN_WIDTH
    if w1 <= SCREEN_WIDTH:
        return w1, SCREEN_HEIGHT
    return SCREEN_WIDTH, h1


def get_frames_number():
    props = imageio.v3.improps(VIDEO_FILENAME)
    # Make sure the codec provides the number of frames
    assert props.shape[0] != -1
    return props.shape[0]


def grayscale_to_palette(img_arr: np.ndarray):
    h, w = img_arr.shape
    counter = 0
    rgb_arr = np.zeros([h, w, RGB_CHANNELS], dtype=np.uint8)
    for pixel in np.nditer(img_arr):
        j = int(counter / w)
        i = int(counter % w)
        grayscale_value = int(pixel)
        rgb_arr[j, i] = PALETTE[grayscale_value]
        counter += 1
    return rgb_arr


def main():
    timer = utils.Timer()
    timer.start()

    # Create the videos folder if it doesn't exist
    if not os.path.exists(VIDEOS_DIR):
        os.mkdir(VIDEOS_DIR)

    counter = 0
    total_frames = get_frames_number()
    step_size = total_frames / 100
    bar = Bar("Processing...", max=100, suffix='%(percent)d%%')
    bar.check_tty = False
    # Read frames from video
    reader = imageio.get_reader(VIDEO_FILENAME)
    writer = imageio.get_writer(
       OUT_VIDEO_FILENAME, format="mp4", mode='I', fps=FPS
    )

    for i, frame in enumerate(reader):
        w0, h0, _ = frame.shape
        frame = np.array(frame) / MAX_COLOR
        grayscale = np.dot(frame, RGB_WEIGHT)
        img_arr = floyd_steinberg_dithering(
            grayscale, add_noise=False, palette=SCALE
        )
        rgb_img = grayscale_to_palette(img_arr)
        writer.append_data(rgb_img)
        counter += 1
        if counter % step_size == 0:
            bar.next()
    bar.finish()
    print("Writing video")
    writer.close()
    timer.stop()
    print(f"Total time spent {timer}")


if __name__ == '__main__':
    main()
