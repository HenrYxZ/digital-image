import numpy as np
from PIL import Image
from progress.bar import Bar

# Local modules
import utils


MAX_QUALITY = 95
COLOR_CHANNELS = 3
MAX_COLOR = 255
V_SAMPLES = 3
H_SAMPLES = 3
TOTAL_SAMPLES = V_SAMPLES * H_SAMPLES

IMG_FILENAME = "mickey.jpg"


def main():
    while True:
        opt = input(
            "Enter an option:\n"
            "[1] for scale\n"
            "[2] for translate\n"
            "[3] for rotate\n"
            "[4] for shear\n"
            "[5] for perspective\n"
            "[6] for mirror\n"
            "[7] for translate + scale\n"
            "[0] to quit\n"
        )
        if opt == '0':
            quit()
        img = Image.open(IMG_FILENAME)
        img_arr = np.array(img)
        h, w, color_channels = img_arr.shape
        timer = utils.Timer()
        timer.start()
        if opt == '1':
            print("Scaling image...")
            scale = 2
            inverse_matrix = np.array([
                [1 / scale, 0, 0],
                [0, 1 / scale, 0],
                [0, 0, 1]
            ])
            new_h = int(scale * h)
            new_w = int(scale * w)
        elif opt == '2':
            print("Translating image...")
            translate_x = 150
            translate_y = 50
            inverse_matrix = np.array([
                [1, 0, -translate_x],
                [0, 1, -translate_y],
                [0, 0, 1]
            ])
            new_h = h + abs(translate_y)
            new_w = w + abs(translate_x)
        elif opt == '3':
            print("Rotating image...")
            theta = np.pi / 6
            new_h = h + np.abs(np.ceil(np.sin(theta) * w)).astype(int)
            new_w = w + np.abs(np.ceil(np.sin(theta) * h)).astype(int)
            inverse_matrix = np.array([
                [np.cos(-theta), np.sin(-theta), 0],
                [-np.sin(-theta), np.cos(-theta), 0],
                [0,                 0,          1]
            ])
        elif opt == '4':
            print("Shearing image...")
            x = 10
            y = 10
            new_h = h + y
            new_w = w + x
            inverse_matrix = np.array([
                [1, -x, 0],
                [-y, 1, 0],
                [0, 0, 1 - x * y]
            ])
        elif opt == '5':
            # Using Pseudo-Code from TA
            print("Applying perspective to image...")
            # a = 150
            # b = 50
            # new_h = h + a + 50
            # new_w = w + b + 50
            # inverse_matrix = np.array([
            #     [1, 0, 0],
            #     [0, 1, 0],
            #     [-a, -b, 1]
            # ])
            # init
            output = np.zeros([h, w, color_channels], dtype=np.uint8)
            # choose four points in image
            x0 = 10 / w
            x1 = 89 / w
            x2 = 150 / w
            x3 = 347 / w
            y0 = 25 / h
            y1 = 159 / h
            y2 = 345 / h
            y3 = 499 / h
            # calculate a0, b0, a1, b1, a2, b2, a3,
            a0 = x0
            a1 = x3 - x0
            a2 = x1 - x0
            a3 = x2 - x1 - x3 - x0
            b0 = y0
            b1 = y3 - y0
            b2 = y1 - y0
            b3 = y2 - y1 - y3 - y0
            for j in range(h):
                for i in range(w):
                    x = i / w
                    y = j / h
                    c0 = a1 * (b0 - y) + b1 * (x - a0)
                    c1 = a3 * (b0 - y) + b3 * (x - a0) + a1 * b2 - a2 * b1
                    c2 = a3 * b2 - a2 * b3
                    v = (-c1) / (2 * c2) + np.sqrt(
                        np.square(c1) - 4 * c2 * c0
                    ) / (2 * c2)
                    u = (x - a0 - a2 * v) / (a1 + a3 * v)
                    if 0 < u < 1 and 0 < v < 1:
                        j1 = int(v * h)
                        i1 = int(u * w)
                        output[j][i] = img_arr[j1][i1]
            timer.stop()
            print(f"Total time spent: {timer}")
            output_img = Image.fromarray(output)
            output_img.save("output.jpg", quality=MAX_QUALITY)
            print(f"Image saved in output.jpg")
            return
        elif opt == '6':
            print("Mirroring image...")
            inverse_matrix = np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            new_h = h
            new_w = w
            output_arr = np.zeros((new_h, new_w, color_channels),
                                  dtype=np.uint8)
            step_size = np.ceil(new_w * new_h / 100).astype('int')
            counter = 0
            bar = Bar('Processing', max=100)
            bar.check_tty = False
            for j in range(new_h):
                for i in range(new_w):
                    color = np.zeros(color_channels)
                    # Samples
                    for n in range(V_SAMPLES):
                        for m in range(H_SAMPLES):
                            r0, r1 = np.random.random_sample(2)
                            x = i + (m + r0) / H_SAMPLES + 0.5
                            y = new_h - 1 - j + (n + r1) / V_SAMPLES + 0.5
                            transform = np.dot(
                                inverse_matrix, np.array([x, y, 1])
                            ).astype(int)
                            if transform[2] != 1:
                                transform = transform / transform[2]
                            u, v = transform[:2]
                            if not (u >= w or v >= h):
                                # sample_color = utils.blerp(img_arr, u, v)
                                sample_color = img_arr[h - 1 - int(v)][int(u)]
                                color += sample_color
                    color = color / TOTAL_SAMPLES
                    output_arr[j][i] = color
                    counter += 1
                    if counter % step_size == 0:
                        bar.next()
            bar.finish()
            timer.stop()
            print(f"Total time spent: {timer}")
            output_img = Image.fromarray(output_arr)
            output_img.save("output.jpg", quality=MAX_QUALITY)
            print(f"Image saved in output.jpg")
            return
        elif opt == '7':
            print("Translate + Scale...")
            inverse_matrix = np.array([
                [2, 0, -256],
                [0, 2, -256],
                [0, 0, 1]
            ])
            new_h = h
            new_w = w
        else:
            inverse_matrix = np.identity(3)
            new_h = h
            new_w = w
        # Apply transformation
        output_arr = np.zeros((new_h, new_w, color_channels), dtype=np.uint8)
        step_size = np.ceil(new_w * new_h / 100).astype('int')
        counter = 0
        bar = Bar('Processing', max=100)
        bar.check_tty = False
        for j in range(new_h):
            for i in range(new_w):
                color = np.zeros(color_channels)
                # Samples
                for n in range(V_SAMPLES):
                    for m in range(H_SAMPLES):
                        r0, r1 = np.random.random_sample(2)
                        x = i + (m + r0) / H_SAMPLES + 0.5
                        y = new_h - 1 - j + (n + r1) / V_SAMPLES + 0.5
                        transform = np.dot(
                            inverse_matrix, np.array([x, y, 1])
                        ).astype(int)
                        if transform[2] != 1:
                            transform = transform / transform[2]
                        u, v = transform[:2]
                        if not (u >= w or v >= h or u < 0 or v < 0):
                            # sample_color = utils.blerp(img_arr, u, v)
                            sample_color = img_arr[h - 1 - int(v)][int(u)]
                            color += sample_color
                color = color / TOTAL_SAMPLES
                output_arr[j][i] = color
                counter += 1
                if counter % step_size == 0:
                    bar.next()
        bar.finish()
        timer.stop()
        print(f"Total time spent: {timer}")
        output_img = Image.fromarray(output_arr)
        output_img.save("output.jpg", quality=MAX_QUALITY)
        print(f"Image saved in output.jpg")


if __name__ == '__main__':
    main()
