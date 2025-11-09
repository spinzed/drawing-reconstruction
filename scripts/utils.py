import json
import numpy as np
import matplotlib.pylab as plt

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

def compile_img(strokes, shape=(256,256)):
    img = np.ones(shape)

    # draw each stroke
    for stroke in strokes:
        xs = stroke[0]
        ys = stroke[1]

        final_xs = []
        final_ys = []

        current_x = xs[0]
        current_y = ys[0]

        # each stroke is composed of multiple registered key points, interpolate between each one of them
        for i in range(1, len(xs)):
            out_xs, out_ys = lerp(current_x, current_y, xs[i], ys[i])
            final_xs += out_xs
            final_ys += out_ys
            current_x = xs[i]
            current_y = ys[i]

        img[final_xs, final_ys] = 0
    return img

# for debugging
if __name__ == "__main__":
    a = load_ndjson("quickdraw/cat.ndjson")
    strokes = a[0]["drawing"]
    img = compile_img(strokes)

    plt.imshow(img, cmap="grey")
    plt.show()
