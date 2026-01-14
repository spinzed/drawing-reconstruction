import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QTextEdit
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal
import numpy as np
import time
import os
import utils
import loader

binarize = False
img_shape = (256, 256)

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
    on_mousedown = Signal(int, int)
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
        pm_w, pm_h = img_shape # TODO: fix properly

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
                self.on_mousedown.emit(pos[0], pos[1])

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

    def registerOnMousedown(self, func):
        self.on_mousedown.connect(func)

    def registerOnMouseup(self, func):
        self.on_release.connect(func)

    def registerWhileDown(self, func):
        self.hover_while_pressed.connect(func)

"""
Main app
"""
class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.loader = loader.Loader(img_shape)
        self.generator = None
        self.category = None
        self.setWindowTitle("Generated Image Viewer")

        # --- Dropdowns ---
        self.weights_dropdown = QComboBox()

        self.cat_dropdown = QComboBox()
        self.cat_dropdown.currentTextChanged.connect(self.on_category_change)

        self.weight_options = get_weights_files()
        self.weights_dropdown.addItems(self.weight_options)
        self.weights_dropdown.setCurrentIndex(-1)
        self.weights_dropdown.currentTextChanged.connect(self.on_weight_file_change)

        # --- Image labels ---
        self.image1 = InteractiveImage()
        self.image1.registerOnMousedown(self.on_canvas_mousedown)
        self.image1.registerWhileDown(self.on_canvas_mousemove)
        self.image1.registerOnMouseup(self.generate)

        self.image2 = QLabel()

        for img in (self.image1, self.image2):
            img.setAlignment(Qt.AlignCenter)
            img.setFixedSize(300, 300)
            img.setStyleSheet("border: 1px solid gray")

        self.reset_canvas()

        # --- Buttons ---
        self.redraw_button = QPushButton(text="Redraw")
        self.redraw_button.clicked.connect(self.on_redraw)

        self.reset_button = QPushButton(text="Reset")
        self.reset_button.clicked.connect(self.on_reset)

        # --- Layouts ---
        dropdowns_layout = QHBoxLayout()
        dropdowns_layout.addWidget(self.weights_dropdown)
        dropdowns_layout.addWidget(self.cat_dropdown)

        images_layout = QHBoxLayout()
        images_layout.addWidget(self.image1)
        images_layout.addWidget(self.image2)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.redraw_button)
        buttons_layout.addWidget(self.reset_button)

        # info area below the buttons (static info/help text)
        self.info_area = QTextEdit()
        self.info_area.setReadOnly(True)
        self.info_area.setPlainText(
            "Info:\n- Draw on the left canvas using the mouse.\n- Click 'Redraw' to generate a reconstruction.\n- Click 'Reset' to clear the canvas.\n- Load weights via the dropdown to enable a generator."
        )
        self.info_area.setFixedHeight(120)
        self.info_area.setStyleSheet("background: #f5f5f5;")

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(dropdowns_layout)
        main_layout.addLayout(images_layout)
        main_layout.addLayout(buttons_layout)
        main_layout.addWidget(self.info_area)

    # --- Handlers ---
    def on_category_change(self, cat):
        print(f"Selected category: {cat}")
        self.category = cat

    def on_weight_file_change(self, file):
        msg = f"Trying to load weights: {file}"
        print(msg)
        self.set_info(msg)
        try:
            self.generator = self.loader.load_from_checkpoint(file)
        except RuntimeError as e:
            err = f"Failed to load weights: {e}"
            print(err)
            self.set_info(err)
            return
        print(f"Weights successfuly set: {file}")
        params = "\n".join([f"    - {k}: {v}" for k, v in self.generator.additional_params().items()])
        self.set_info(f"Model: {self.generator.model_type}\nSupported categories: {', '.join(self.generator.supported_classes)}\nModel params:\n{params}")

        self.cat_dropdown.clear()
        self.cat_dropdown.addItems(self.generator.supported_classes)
        self.cat_dropdown.setCurrentIndex(0)

    def on_canvas_mousedown(self, x, y):
        self.strokes.append([[x], [y]])

    def on_canvas_mousemove(self, x, y, xlast, ylast):
        xs, ys = utils.lerp(xlast, ylast, x, y)
        self.canvas[ys, xs] = 0
        self.set_canvas(self.canvas)
        self.strokes[-1][0].append(x)
        self.strokes[-1][1].append(y)

    def on_redraw(self):
        self.generate()

    def on_reset(self):
        self.reset_canvas()

    # --- Setters ---
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

    def set_info(self, text: str):
        try:
            self.info_area.setPlainText(text)
        except Exception:
            print(text)

    # --- Actions ---
    def reset_canvas(self):
        empty = (np.ones(img_shape) * 255).astype(np.uint8)
        self.canvas = empty
        self.strokes = []
        self.set_canvas(self.canvas)
        self.set_generated(empty)

    def generate(self):
        if len(self.strokes) == 0:
            print("Canvas empty, skipping generation")
            return
        if self.generator is None:
            print("No valid generator loaded")
            return

        float_img = (self.canvas / 255).astype(np.float32)
        generated = self.generator.generate(float_img, self.strokes)
        self.set_generated((generated * 255).astype(np.uint8))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ImageApp()
    w.show()
    sys.exit(app.exec())
