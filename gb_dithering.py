import av
from numba import njit
import numpy as np
import os.path
from tqdm import tqdm

# Local Modules
from constants import RGB_CHANNELS
from dithering import floyd_steinberg_dithering_njit
import utils
from utils import scale_nn_njit


VIDEOS_DIR = "videos"
# VIDEO_FILENAME = f"{VIDEOS_DIR}/anim_final_raytraced.mp4"
# VIDEO_FILENAME = f"{VIDEOS_DIR}/miri.mp4"
VIDEO_FILENAME = f"{VIDEOS_DIR}/gta.mp4"
GRAYSCALE_FILENAME = f"{VIDEOS_DIR}/grayscale.mp4"
RESIZED_FILENAME = f"{VIDEOS_DIR}/resized.mp4"
DITHERED_FILENAME = f"{VIDEOS_DIR}/dithered.mp4"
SMALL_FILENAME = f"{VIDEOS_DIR}/small.mp4"
OUT_VIDEO_FILENAME = f"{VIDEOS_DIR}/out.mp4"
MAX_QUALITY = 95
FPS = 12
SCALE = (0.0, 0.33, 0.66, 1.0)
PALETTE = np.array(
    ((41, 65, 57), (57, 89, 74), (90, 121, 66), (123, 130, 16)),
    dtype=np.uint8
)
GRAYSCALE_PALETTE = np.array([0, 84, 168, 255], dtype=np.uint8)
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
    if h1 > SCREEN_HEIGHT:
        return SCREEN_WIDTH, h1
    return w1, SCREEN_HEIGHT


@njit
def grayscale_to_palette(img_arr: np.ndarray) -> np.ndarray:
    h, w = img_arr.shape
    rgb_arr = np.zeros((h, w, RGB_CHANNELS), dtype=np.uint8)

    for i, grayscale_color in enumerate(GRAYSCALE_PALETTE):
        mask = img_arr == grayscale_color
        mask_rgb = np.stack((mask, mask, mask), axis=-1)
        rgb_arr = rgb_arr + mask_rgb * PALETTE[i]

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
    container = av.open(VIDEO_FILENAME)
    total_frames = container.streams.video[0].frames

    h_final = SCREEN_HEIGHT * PIXEL_SIZE
    w_final = SCREEN_WIDTH * PIXEL_SIZE

    out_container = av.open(OUT_VIDEO_FILENAME, mode="w")

    out_stream = out_container.add_stream(
        codec_name="libx264",
        rate=container.streams.video[0].average_rate.numerator
    )
    out_stream.width = w_final
    out_stream.height = h_final
    out_stream.pix_fmt = "yuv420p"

    has_audio = len(container.streams.audio) > 0
    if has_audio:
        src_audio = container.streams.audio[0]
        out_audio = out_container.add_stream_from_template(src_audio)

    for i, frame in tqdm(
        enumerate(container.decode(video=0)),
        desc="Reading Video",
        total=total_frames
    ):
        # Resize to fit Game Boy screen
        # -------------------------------------------------------------------------
        w0 = frame.width
        h0 = frame.height
        w1, h1 = fit_screen(w0, h0)

        # --------------------------------------------------------------------------
        # Normalize & dither
        im_arr = frame.reformat(width=w1, height=h1, format='gray').to_ndarray()
        if h1 > SCREEN_HEIGHT:
            vertical_offset = (h1 - SCREEN_HEIGHT) // 2
            horizontal_offset = 0
        else:
            vertical_offset = 0
            horizontal_offset = (w1 - SCREEN_WIDTH) // 2
        vertical_limit = vertical_offset + SCREEN_HEIGHT
        horizontal_limit = horizontal_offset + SCREEN_WIDTH
        cropped_arr = im_arr[
            vertical_offset:vertical_limit,
            horizontal_offset:horizontal_limit
        ]
        dithered = floyd_steinberg_dithering_njit(
            cropped_arr, GRAYSCALE_PALETTE
        )

        # -------------------------------------------------------------------------
        # Colorize with Game Boy palette
        rgb_img_arr = grayscale_to_palette(dithered)

        # Scale up the frame with nearest neighbor
        final_arr = scale_nn_njit(rgb_img_arr, h_final, w_final)

        # write frame into stream
        video_frame = av.VideoFrame.from_ndarray(final_arr, format="rgb24")
        for packet in out_stream.encode(video_frame):
            out_container.mux(packet)

    # Flush stream
    for packet in out_stream.encode():
        out_container.mux(packet)

    if has_audio:
        container.seek(0)
        for packet in container.demux(src_audio):
            if packet.dts is None:
                continue

            # Reassign the packet to the output audio stream
            packet.stream = out_audio

            # Mux the audio packet directly
            out_container.mux(packet)

    # -------------------------------------------------------------------------
    timer.stop()
    print(f"Total time spent {timer}")
    print(f"Output video saved in {OUT_VIDEO_FILENAME}")

    container.close()
    out_container.close()

if __name__ == '__main__':
    main()
