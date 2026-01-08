import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal
import numpy as np
import time
import os
import utils
from generator import VaeGenerator, ConvVaeGenerator

def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    if arr.dtype != np.uint8:
        raise ValueError("Array must be uint8")

    if arr.ndim == 2:
        h, w = arr.shape
        bytes_per_line = w
        qimg = QImage(
            arr.data, w, h, bytes_per_line, QImage.Format_Grayscale8
        )

    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        bytes_per_line = 3 * w
        qimg = QImage(
            arr.data, w, h, bytes_per_line, QImage.Format_RGB888
        )

    else:
        raise ValueError("Unsupported shape")

    return QPixmap.fromImage(qimg)

# returns dict category: file path
def get_categories():
    files = os.listdir("quickdraw")
    ext = "ndjson"
    categories = {file.split("." + ext)[0]: file for file in files if file.endswith(ext)}
    return categories

def get_weights_files():
    files = os.listdir(".")
    ext = "pth"
    weight_files = [file for file in files if file.endswith(ext)]
    return weight_files


"""
Helper class for images.
"""
class InteractiveImage(QLabel):
    hover_while_pressed = Signal(int, int, int, int)
    on_release = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.pressed = False
        self.func = None
        self.last_time = time.time()
        self.last_coords = None

    def _map_to_image(self, event):
        if not self.pixmap():
            return None

        label_w, label_h = self.width(), self.height()
        #pixmap = self.pixmap()
        #pm_w, pm_h = pixmap.width(), pixmap.height()
        pm_w, pm_h = 256, 256 # TODO: fix properly

        scale = min(label_w / pm_w, label_h / pm_h)
        drawn_w = pm_w * scale
        drawn_h = pm_h * scale

        offset_x = (label_w - drawn_w) / 2
        offset_y = (label_h - drawn_h) / 2

        x = event.position().x() - offset_x
        y = event.position().y() - offset_y

        if 0 <= x < drawn_w and 0 <= y < drawn_h:
            return int(x / scale), int(y / scale)

        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self._map_to_image(event)
            if pos:
                self.pressed = True

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            #pos = self._map_to_image(event)
            self.pressed = False
            self.last_coords = None
            self.on_release.emit()

    def mouseMoveEvent(self, event):
        pos = self._map_to_image(event)
        t = time.time()
        if pos and self.pressed and t - self.last_time > 1/60:
            self.last_time = t
            self._emitHover(*pos)

    def _emitHover(self, x, y):
        if self.last_coords is None:
            self.last_coords = (x, y)
        last = self.last_coords
        self.last_coords = (x, y)
        self.hover_while_pressed.emit(x, y, last[0], last[1])

    def registerOnMouseup(self, func):
        self.on_release.connect(func)

    def registerWhileDown(self, func):
        def wrapped_func(x, y):
            return func(x, y, last[0], last[1])

        self.hover_while_pressed.connect(func)

""".
Main. app
"""
class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.generator = ConvVaeGenerator()
        self.setWindowTitle("Generated Image Viewer")

        # --- Dropdowns ---
        self.cat_dropdown = QComboBox()
        self.cat_dropdown.addItems(get_categories())
        self.cat_dropdown.currentTextChanged.connect(self.on_category_change)

        self.weights_dropdown = QComboBox()
        self.weight_options = get_weights_files()
        self.weights_dropdown.addItems(self.weight_options)
        if len(self.weight_options) > 0:
            self.generator.set_weights(self.weight_options[0])
        self.weights_dropdown.currentTextChanged.connect(self.on_weight_file_change)

        # --- Image labels ---
        self.image1 = InteractiveImage()
        self.image1.registerWhileDown(self.on_canvas_click)
        self.image1.registerOnMouseup(self.generate)

        self.image2 = QLabel()

        for img in (self.image1, self.image2):
            img.setAlignment(Qt.AlignCenter)
            img.setFixedSize(300, 300)
            img.setStyleSheet("border: 1px solid gray")

        self.reset_canvas()

        # --- Buttons ---
        self.reset_button = QPushButton(text="Reset")
        self.reset_button.clicked.connect(self.on_reset)

        # --- Layouts ---
        dropdowns_layout = QHBoxLayout()
        dropdowns_layout.addWidget(self.cat_dropdown)
        dropdowns_layout.addWidget(self.weights_dropdown)

        images_layout = QHBoxLayout()
        images_layout.addWidget(self.image1)
        images_layout.addWidget(self.image2)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(dropdowns_layout)
        main_layout.addLayout(images_layout)
        main_layout.addWidget(self.reset_button)

    # --- Handlers ---
    def set_canvas(self, img: np.ndarray):
        pixmap = numpy_to_qpixmap(img)
        self.image1.setPixmap(pixmap.scaled(
            300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def set_generated(self, img: np.ndarray):
        pixmap = numpy_to_qpixmap(img)
        self.image2.setPixmap(pixmap.scaled(
            300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    # does nothing as of now
    def on_category_change(self, cat):
        print(f"Selected cat: {cat}")

    def on_weight_file_change(self, w):
        print(f"Selected weights: {w}")
        self.generator.set_weights(w)

    def on_canvas_click(self, x, y, xlast, ylast):
        #print(f"Click ({x}, {y}), ({xlast}, {ylast})")
        xs, ys = utils.lerp(xlast, ylast, x, y)
        self.canvas[ys, xs] = 0
        self.set_canvas(self.canvas)

    def on_reset(self):
        self.reset_canvas()
    
    def reset_canvas(self):
        empty = (np.ones((256, 256)) * 255).astype(np.uint8)
        self.canvas = empty
        self.set_canvas(self.canvas)
        self.set_generated(empty)

    def generate(self):
        if np.sum(self.canvas) == 0: # canvas is empty, don't attempt to reconstruct
            return
        generated = self.generator.generate((self.canvas/ 255).astype(np.float32))
        self.set_generated((generated * 255).astype(np.uint8))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ImageApp()
    w.show()
    sys.exit(app.exec())
