import numpy as np
import os.path


DEFAULT_MAX_COLOR = 255
DEFAULT_SEPARATOR = ' '
COMMENT_CHAR = '#'
NUMBER_OF_CHANNELS = 3
ENCODING = "ascii"
MAGIC_NUMBER = "P3"


def read_line(f):
    """
    Read line ignoring comments.

    f(File): The file that you are reading

    str: Returns the string with the line
    """
    line = f.readline()
    while line[0] == COMMENT_CHAR:
        line = f.readline()
    return line


def read_image(f, w, h):
    """
    Read the image from this file.
    f(File): The file that you are reading, pointing to the data lines
    w(int): Width of the Image
    h(int): Height of the Image

    array: A numpy array with the image data
    """
    img_arr = np.zeros([h, w, NUMBER_OF_CHANNELS], dtype=np.uint8)
    current_row = 0
    current_column = 0
    for line in f.readlines():
        num_str = line.split(DEFAULT_SEPARATOR)
        # don't take the last elemnt which is \n
        num_str = num_str[:len(num_str) - 1]
        colors = [
            num_str[i:i + NUMBER_OF_CHANNELS] for i in range(
                0, len(num_str), NUMBER_OF_CHANNELS
            )
        ]
        for color in colors:
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            img_arr[current_row][current_column] = [r, g, b]
            current_column += 1
            if current_column == w:
                current_column = 0
        current_row += 1
    return img_arr


class PPMFile:
    def __init__(self, filename):
        self.filename = filename
        if os.path.exists(filename):
            with open(filename, mode="r", encoding="ascii") as f:
                self.magic_number = read_line(f)
                width_height = read_line(f).split(DEFAULT_SEPARATOR)
                self.width = int(width_height[0])
                self.height = int(width_height[1])
                self.max_color = int(read_line(f))
                self.data = read_image(f, self.width, self.height)
        else:
            self.magic_number = MAGIC_NUMBER
            self.max_color = DEFAULT_MAX_COLOR

    def from_array(self, img_arr):
        if len(img_arr.shape) == 2:
            self.height, self.width = img_arr.shape
        else:
            self.height, self.width, _ = img_arr.shape
        self.data = img_arr

    def save(self, filename=None):
        if not filename:
            filename = self.filename
        with open(filename, mode="w", encoding="ascii") as f:
            f.write(f"{self.magic_number}\n")
            f.write(f"{self.width} {self.height}\n")
            f.write(f"{self.max_color}\n")
            print("Working...")
            for j in range(self.height):
                for i in range(self.width):
                    for k in range(NUMBER_OF_CHANNELS):
                        f.write(f"{self.data[j][i][k]} ")
                f.write('\n')
