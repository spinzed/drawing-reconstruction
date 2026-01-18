import cv2
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QTextEdit, QCheckBox, QSpinBox, QSlider
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal
import numpy as np
import time
import os
import sys
import data.utils as utils
import data.loader as loader

binarize = False
canvas_shape = (64, 64)
weights_dir = "weights"

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
    if not os.path.isdir("quickdraw"):
        print("Category folder 'quickdraw' not found")
        return []

    files = os.listdir("quickdraw")
    ext = "ndjson"
    categories = {file.split("." + ext)[0]: file for file in files if file.endswith(ext)}
    return categories

def get_weight_files():
    if not os.path.isdir(weights_dir):
        print(f"Weights folder '{weights_dir}' not found")
        return []

    files = os.listdir(weights_dir)
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
        pm_w, pm_h = canvas_shape # TODO: fix properly

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
        self.loader = loader.Loader(image_size=canvas_shape)
        self.generator = None
        self.category = None
        self.setWindowTitle("Generated Image Viewer")

        # --- Dropdowns ---
        self.weights_dropdown = QComboBox()

        self.cat_dropdown = QComboBox()
        self.cat_dropdown.currentTextChanged.connect(self.on_category_change)

        self.weight_options = get_weight_files()
        self.weights_dropdown.addItems(self.weight_options)
        self.weights_dropdown.setCurrentIndex(-1)
        self.weights_dropdown.currentTextChanged.connect(self.on_weight_file_change)

        # --- Output filter controls ---
        self.canny_checkbox = QCheckBox("Canny")
        self.canny_down = QSpinBox()
        self.canny_up = QSpinBox()
        self.canny_down.setMinimum(0)
        self.canny_down.setMaximum(255)
        self.canny_down.setValue(100)
        self.canny_up.setMinimum(0)
        self.canny_up.setMaximum(255)
        self.canny_up.setValue(200)
        self.canny_checkbox.stateChanged.connect(self.on_output_filter_change)
        self.canny_down.valueChanged.connect(self.on_output_filter_change)
        self.canny_up.valueChanged.connect(self.on_output_filter_change)

        self.binarize_checkbox = QCheckBox("Binarize")
        self.binarize_checkbox.stateChanged.connect(self.on_output_filter_change)

        self.binarize_threshold = QSlider(Qt.Horizontal)
        self.binarize_threshold.setMinimum(0.0)
        self.binarize_threshold.setMaximum(100.0)
        self.binarize_threshold.setValue(50.0)
        self.binarize_threshold.valueChanged.connect(self.on_output_filter_change)

        self.threshold_label = QLabel("--")
        self.threshold_label.setFixedWidth(40)

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

        #self.dataset = np.load("quickdraw/alarm clock.npz", encoding="latin1", allow_pickle=True)
        #data = dataset["train"]
        #sequence = data[12832]
        #print(sequence)
        #self.set_canvas((utils.compile_img_from_sequence(sequence, img_shape=(512, 512)) * 255).astype(np.uint8))

        # --- Buttons ---
        self.redraw_button = QPushButton(text="Regenerate")
        self.redraw_button.clicked.connect(self.on_regenerate)

        self.reset_button = QPushButton(text="Reset")
        self.reset_button.clicked.connect(self.on_reset)

        # --- Layouts ---
        dropdowns_layout = QHBoxLayout()
        dropdowns_layout.addWidget(self.weights_dropdown)
        dropdowns_layout.addWidget(self.cat_dropdown)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(self.canny_checkbox)
        filter_layout.addWidget(self.canny_down)
        filter_layout.addWidget(self.canny_up)
        filter_layout.addWidget(self.binarize_checkbox)
        filter_layout.addWidget(self.binarize_threshold)
        filter_layout.addWidget(self.threshold_label)

        images_layout = QHBoxLayout()
        images_layout.addWidget(self.image1)
        images_layout.addWidget(self.image2)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.redraw_button)
        buttons_layout.addWidget(self.reset_button)

        # -- Info Area (static info/help text) --
        self.info_area = QTextEdit()
        self.info_area.setReadOnly(True)
        self.info_area.setPlainText(
            "Info:\n- Draw on the left canvas using the mouse.\n- Click 'Regenerate' to regenerate a reconstruction.\n- Click 'Reset' to clear the canvas.\n- Load weights via the dropdown to enable a generator."
        )
        self.info_area.setFixedHeight(120)
        self.info_area.setStyleSheet("background: #f5f5f5;")

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(dropdowns_layout)
        main_layout.addLayout(filter_layout)
        main_layout.addLayout(images_layout)
        main_layout.addLayout(buttons_layout)
        main_layout.addWidget(self.info_area)

    # --- Handlers ---
    def on_category_change(self, cat):
        print(f"Selected category: {cat}")
        self.category = cat

    def on_output_filter_change(self):
        canny_checked = self.canny_checkbox.isChecked()
        down = self.canny_down.value()
        up = self.canny_up.value()
        binarize_checked = self.binarize_checkbox.isChecked()
        threshold = self.binarize_threshold.value() / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
        print(f"Output filter changed: Canny(enabled={canny_checked}, down={down}, up={up}), Binarize(enabled={binarize_checked}, threshold={threshold:.2f})")
        self.set_generated(self.generated_image)

    def on_weight_file_change(self, filename):
        file = os.path.join(weights_dir, filename)
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
        info = f"Model: {self.generator.model_type}\nSupported categories: {', '.join(self.generator.supported_classes)}"
        if self.generator.checkpoint.get("epoch") is not None:
            info += f"\nTrained epochs: {self.generator.checkpoint['epoch']}"
        if self.generator.checkpoint.get('train_images') is not None:
            info += f"\nNumber of images in train set: {self.generator.checkpoint['train_images']}"
        info += f"\nModel params:\n{params}"
        self.set_info(info)

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

    def on_regenerate(self):
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
        self.generated_image = img

        img = img.copy()

        # interpolate from model size to canvas size
        if img.shape != canvas_shape:
            img = cv2.resize(img, canvas_shape, interpolation=cv2.INTER_LINEAR)

        # filters
        if self.canny_checkbox.isChecked():
            low = self.canny_down.value()
            high = self.canny_up.value()
            img = np.subtract(255, cv2.Canny((img * 255).astype(np.uint8), low, high))

        if self.binarize_checkbox.isChecked():
            threshold = (self.binarize_threshold.value() / 100.0) * 256.0
            img[img >= threshold] = 255
            img[img < threshold] = 0

        pixmap = numpy_to_qpixmap(img) # img must be uint8
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
        empty = (np.ones(canvas_shape) * 255).astype(np.uint8)
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

        print("Generating...")

        generated = self.generator.generate(self.canvas, self.strokes)
        self.set_generated((generated * 255).astype(np.uint8))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ImageApp()
    w.show()
    sys.exit(app.exec())
