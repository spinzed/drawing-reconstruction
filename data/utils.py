import json
import numpy as np
import matplotlib.pylab as plt
import cv2

drawings = []

def load_ndjson(name):
    data = []
    with open(name, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def lerp(x1, y1, x2, y2):
    """
    Rasterize a line from (x1, y1) to (x2, y2) using Bresenham's algorithm.
    Returns a list of (x, y) coordinate tuples representing the line.
    """
    xs = []
    ys = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Determine direction of line
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    err = dx - dy

    x, y = x1, y1

    while True:
        xs.append(x)
        ys.append(y)

        # Check if we've reached the end point
        if x == x2 and y == y2:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x += sx

        if e2 < dx:
            err += dx
            y += sy

    return xs, ys

def compile_img_from_strokes(strokes, img_shape=(256, 256), img=None, start=0, end=None, pad=10):
    assert end is None or end <= len(strokes), "end must be smaller or equal to list size"

    if img is None:
        img = np.ones(img_shape)

    scale_x = (img_shape[1] - pad) / 256
    scale_y = (img_shape[0] - pad) / 256

    # draw each stroke
    for stroke_ind in range(start, end if end is not None else len(strokes)):
        stroke = strokes[stroke_ind]
        xs = stroke[0]
        ys = stroke[1]

        final_xs = []
        final_ys = []

        # scaling of points
        current_x = int(xs[0] * scale_y) + pad // 2
        current_y = int(ys[0] * scale_x) + pad // 2

        # each stroke is composed of multiple registered key points, interpolate between each one of them
        for i in range(1, len(xs)):
            xs_i = int(scale_x * xs[i])
            ys_i = int(scale_y * ys[i])
            out_xs, out_ys = lerp(current_x, current_y, xs_i, ys_i)
            final_xs += out_xs
            final_ys += out_ys
            current_x = xs_i
            current_y = ys_i

        #final_ys = [int(round(y * shape[0] / 256)) for y in final_ys]
        #final_xs = [int(round(x * shape[0] / 256)) for x in final_xs]
        img[final_ys, final_xs] = 0
    return img

"""
Sequence is an array-like of shape [seq_length, [dx, dy, stroke_end(bool)]]
"""
def compile_img_from_sequence(sequence, relative_offsets=True, img_shape=(256, 256), img=None, pad=10):
    if img is None:
        img = np.ones(img_shape)

    scale_x = (img_shape[1] - pad) / 256
    scale_y = (img_shape[0] - pad) / 256

    x, y = img_shape[1]//2 + sequence[0, 1], img_shape[0]//2 + sequence[0, 0]
    #color = np.random.uniform(size=3)
    last = False
    for i in range(1, len(sequence)):
        entry = sequence[i]
        if relative_offsets:
            x2 = x + entry[1]
            y2 = y + entry[0]
        else:
            x2 = entry[1]
            y2 = entry[0]

        if not last:
            xs, ys = lerp(x, y, x2, y2)
            xs = np.clip(xs, 0, img_shape[1]-1)
            ys = np.clip(ys, 0, img_shape[0]-1)
            xs = np.floor(xs * scale_x).astype(np.uint8) + pad // 2
            ys = np.floor(ys * scale_y).astype(np.uint8) + pad // 2
            img[xs, ys] = 0
        last = False

        if entry[2] == 1:
            last = True

        if relative_offsets:
            x += entry[1]
            y += entry[0]
        else:
            x = entry[1]
            y = entry[0]

    return img # returns float image with binary values (0, 1)

def compile2(sequence, img_shape=(256, 256), img=None):
    if img is None:
        img = np.ones(img_shape, dtype=np.float32)

    # start near center (important, otherwise most sketches go out of frame)
    y = img_shape[0] // 2
    x = img_shape[1] // 2

    for i in range(len(sequence)):
        dy, dx, pen_up = sequence[i]

        new_y = y + dy
        new_x = x + dx

        # draw only if pen is DOWN
        if pen_up == 0:
            xs, ys = lerp(x, y, new_x, new_y)
            xs = np.clip(xs, 0, img_shape[1] - 1)
            ys = np.clip(ys, 0, img_shape[0] - 1)
            img[ys, xs] = 0

        y, x = new_y, new_x

    return img

def strokes_to_relative_sequence(strokes, offset=None):
    sequence = []
    lasty, lastx = 0, 0
    if offset is not None:
        lasty, lastx = offset
    for stroke in strokes:
        xs = stroke[0]
        ys = stroke[1]

        for i in range(len(xs)):
            sequence.append([xs[i] - lastx, ys[i] - lasty, 0])
            lasty, lastx = ys[i], xs[i]
            sequence[-1][2] = 0

        sequence[-1][2] = 1
    return np.array(sequence)

def scale_sequence_to_size(sequence, size=256):
    smin = np.min(np.abs(sequence[:, 0:2]))
    smax = np.max(np.abs(sequence[:, 0:2]))

    s2 = sequence.copy()
    s2[:, 0:2] = ((s2[:, 0:2])/(smax-smin)) * size//2 + size//2
    s2 = s2.astype(np.int32)

    return s2

def erode_image(img, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = img.astype(np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)

# for debugging
class CompileTest:
    def __init__(self):
        self.test_data = load_ndjson("quickdraw/cat.ndjson")

    def test(self, index=0, size=100):
        strokes = self.test_data[index]["drawing"]
        print(f"strokes size: {len(strokes)}")
        img = compile_img_from_strokes(strokes, end=min(len(strokes), size))

        plt.imshow(img, cmap="grey")
        plt.show()
