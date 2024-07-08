import imageio.v3 as iio
import numpy as np
import os.path

# Local Modules
from constants import MAX_COLOR, RGB_CHANNELS
from dithering import floyd_steinberg_dithering
import utils
from utils import backwards_mapping, nearest_neighbor_uv


VIDEOS_DIR = "videos"
VIDEO_FILENAME = f"{VIDEOS_DIR}/anim_final_raytraced.mp4"
GRAYSCALE_FILENAME = f"{VIDEOS_DIR}/grayscale.mp4"
RESIZED_FILENAME = f"{VIDEOS_DIR}/resized.mp4"
DITHERED_FILENAME = f"{VIDEOS_DIR}/dithered.mp4"
SMALL_FILENAME = f"{VIDEOS_DIR}/small.mp4"
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
PIXEL_SIZE = 3
RGB_WEIGHT = np.array([0.2989, 0.5870, 0.1140])


def fit_screen(w0: int, h0: int) -> tuple[int, int]:
    """
    Fit the screen by returning new width and height. Calculate the width if the
    screen height is used, and the height if the screen width is used, and use
    the one that fits.
    """
    h1 = int((SCREEN_WIDTH / w0) * h0)
    w1 = int((SCREEN_HEIGHT / h0) * w0)
    if h1 <= SCREEN_HEIGHT:
        return SCREEN_WIDTH, h1
    return w1, SCREEN_HEIGHT


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

    print("Starting the process!")
    # Create the videos folder if it doesn't exist
    if not os.path.exists(VIDEOS_DIR):
        print("Creating video folder")
        os.mkdir(VIDEOS_DIR)

    # Read frames from video
    print(f"Reading video file {VIDEO_FILENAME}")
    frames = iio.imread(VIDEO_FILENAME)
    total_frames, h0, w0, color_channels = frames.shape

    # Resize to fit Game Boy screen
    # -------------------------------------------------------------------------
    print("Resizing video to fit Game Boy")
    w1, h1 = fit_screen(w0, h0)
    resized = np.zeros([total_frames, h1, w1, RGB_CHANNELS])
    for i, frame in enumerate(frames):
        resized[i] = backwards_mapping(frame, h1, w1, utils.blerp_uv)
    iio.imwrite(RESIZED_FILENAME, resized, fps=FPS)

    # Transform to grayscale
    # -------------------------------------------------------------------------
    print("Transforming video to grayscale")
    frames = np.dot(resized, RGB_WEIGHT)
    grayscale = np.round(frames).astype(np.uint8)
    grayscale_as_rgb = np.stack([grayscale] * 3, axis=-1)
    iio.imwrite(GRAYSCALE_FILENAME, grayscale_as_rgb, fps=FPS)

    # --------------------------------------------------------------------------
    # Normalize & dither
    print("Dithering")
    normalized = grayscale.astype(float) / MAX_COLOR
    dithered_rgb = np.zeros([total_frames, h1, w1, RGB_CHANNELS])
    dithered = np.zeros([total_frames, h1, w1], dtype=np.uint8)
    for i, frame in enumerate(normalized):
        # Use dithering to transform 256 grayscale to 4 colors grayscale
        dithered[i] = floyd_steinberg_dithering(
            frame, add_noise=False, palette=SCALE
        )
        dithered_rgb[i] = np.stack([dithered[i]] * 3, axis=-1)
        iio.imwrite(DITHERED_FILENAME, dithered_rgb, fps=FPS)

    # -------------------------------------------------------------------------
    # Colorize with Game Boy palette
    print("Colorizing with the Game Boy palette")
    output_frames = np.ones(
        [total_frames, SCREEN_HEIGHT, SCREEN_WIDTH, color_channels],
        dtype=np.uint8
    ) * PALETTE[0]
    for i, frame in enumerate(dithered):
        # Transform to Game Boy color palette
        rgb_img_arr = grayscale_to_palette(frame)
        if SCREEN_HEIGHT - h1 > 0:
            vertical_offset = int((SCREEN_HEIGHT - h1) / 2)
            horizontal_offset = 0
        else:
            vertical_offset = 0
            horizontal_offset = int((SCREEN_WIDTH - w1) / 2)
        vertical_limit = vertical_offset + h1
        horizontal_limit = horizontal_offset + w1
        output_frames[
            i,
            vertical_offset:vertical_limit,
            horizontal_offset:horizontal_limit
        ] = rgb_img_arr
        iio.imwrite(SMALL_FILENAME, output_frames, fps=FPS)

    # -------------------------------------------------------------------------
    # Scale up the video to be bigger
    print("Scaling up the video")
    h_final = SCREEN_HEIGHT * PIXEL_SIZE
    w_final = SCREEN_WIDTH * PIXEL_SIZE
    final_frames = np.zeros(
        [
            total_frames,
            h_final,
            w_final,
            color_channels
        ],
        dtype=np.uint8
    )
    for i, frame in enumerate(output_frames):
        final_frames[i] = backwards_mapping(
            frame, h_final, w_final, nearest_neighbor_uv
        )

    print("Writing video")
    iio.imwrite(OUT_VIDEO_FILENAME, final_frames, fps=FPS)
    # -------------------------------------------------------------------------
    timer.stop()
    print(f"Total time spent {timer}")


if __name__ == '__main__':
    main()
