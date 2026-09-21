import sys
import json
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QLabel, QPushButton, QSlider,
    QSpinBox, QDoubleSpinBox, QLineEdit, QFileDialog, QHBoxLayout,
    QVBoxLayout, QGridLayout, QGroupBox, QMessageBox, QCheckBox
)


APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "GreenScreenRecolor_settings.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


DEFAULTS = {
    "input_dir": "",
    "output_dir": "",
    "sample_rgb": [21, 146, 52],
    "target_rgb": [0, 0, 0],
    "hue_range": 18.0,
    "sat_min": 45,
    "value_min": 20,
    "value_max": 255,
    "color_distance": 180.0,
    "softness": 8.0,
    "use_rgb_distance": False,
    "show_mask": False,
    "min_area": 0,
    "preserve_luminance": False,
}


def load_settings():
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            result = DEFAULTS.copy()
            result.update(data)
            return result
    except Exception:
        pass
    return DEFAULTS.copy()


def save_settings(data):
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rgb_to_hsv(rgb):
    arr = np.uint8([[rgb]])
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[0, 0]
    return hsv.astype(float)


def circular_hue_distance(h1, h2):
    return np.abs(((h1 - h2 + 90.0) % 180.0) - 90.0)


def make_preview(bgr, sample_rgb, target_rgb, hue_range, sat_min, value_min,
                 value_max, color_distance, softness, min_area,
                 preserve_luminance, use_rgb_distance=False):
    """HSVを主判定にしたグリーンバック検出。各条件を独立したフィルターとして扱う。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sample_h = rgb_to_hsv(sample_rgb)[0]
    hue_dist = circular_hue_distance(hsv[:, :, 0], sample_h)
    mask_binary = (hue_dist <= float(hue_range)) & (hsv[:, :, 1] >= float(sat_min))
    mask_binary &= (hsv[:, :, 2] >= float(value_min)) & (hsv[:, :, 2] <= float(value_max))

    if use_rgb_distance:
        sample = np.array(sample_rgb, dtype=np.float32)
        rgb_dist = np.sqrt(np.sum((rgb.astype(np.float32) - sample) ** 2, axis=2))
        mask_binary &= rgb_dist <= float(color_distance)

    binary = mask_binary.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        keep = np.zeros_like(binary)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 255
        binary = keep

    if softness <= 0:
        mask = binary.astype(np.float32) / 255.0
    else:
        inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        outside = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 3)
        signed = inside - outside
        width = max(float(softness), 0.1)
        mask = np.clip(0.5 + signed / (2.0 * width), 0.0, 1.0)

    out = rgb.astype(np.float32)
    target = np.array(target_rgb, dtype=np.float32)
    if preserve_luminance:
        lum = 0.2126 * out[:, :, 0] + 0.7152 * out[:, :, 1] + 0.0722 * out[:, :, 2]
        target_lum = max(0.2126 * target[0] + 0.7152 * target[1] + 0.0722 * target[2], 1.0)
        replacement = np.clip(target[None, None, :] * (lum[:, :, None] / target_lum), 0, 255)
    else:
        replacement = np.broadcast_to(target, out.shape)
    out = out * (1.0 - mask[:, :, None]) + replacement * mask[:, :, None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR), mask


class ImageViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 420)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#202020; border:1px solid #555;")
        self.image = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.click_callback = None

    def set_image(self, image_rgb):
        self.image = image_rgb
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if self.image is None:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "画像を読み込んでください")
            return

        h, w = self.image.shape[:2]
        scale = min(self.width() / w, self.height() / h)
        dw, dh = int(w * scale), int(h * scale)
        resized = cv2.resize(self.image, (dw, dh), interpolation=cv2.INTER_AREA)
        qimg = QImage(resized.data, dw, dh, dw * 3, QImage.Format_RGB888)
        x = (self.width() - dw) // 2
        y = (self.height() - dh) // 2
        self.scale = scale
        self.offset = (x, y)
        painter.drawImage(x, y, qimg)

    def mousePressEvent(self, event):
        if self.image is None or event.button() != Qt.LeftButton:
            return
        x0, y0 = self.offset
        x = int((event.position().x() - x0) / max(self.scale, 1e-9))
        y = int((event.position().y() - y0) / max(self.scale, 1e-9))
        h, w = self.image.shape[:2]
        if 0 <= x < w and 0 <= y < h and self.click_callback:
            self.click_callback(int(self.image[y, x, 0]), int(self.image[y, x, 1]), int(self.image[y, x, 2]))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.files = []
        self.current_index = 0
        self.original_bgr = None
        self.updating = False
        self.cancel_requested = False

        self.setWindowTitle("Green Screen Recolor")
        self.resize(1200, 900)
        self.build_ui()
        self.load_settings_to_ui()

        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.update_preview)

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        paths = QGroupBox("フォルダ")
        pg = QGridLayout(paths)
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        in_btn = QPushButton("入力フォルダ")
        out_btn = QPushButton("出力フォルダ")
        in_btn.clicked.connect(self.select_input)
        out_btn.clicked.connect(self.select_output)
        pg.addWidget(QLabel("入力:"), 0, 0)
        pg.addWidget(self.input_edit, 0, 1)
        pg.addWidget(in_btn, 0, 2)
        pg.addWidget(QLabel("出力:"), 1, 0)
        pg.addWidget(self.output_edit, 1, 1)
        pg.addWidget(out_btn, 1, 2)
        main.addWidget(paths)

        viewer_group = QGroupBox("プレビュー")
        vg = QVBoxLayout(viewer_group)
        self.viewer = ImageViewer()
        self.viewer.click_callback = self.pick_color
        vg.addWidget(self.viewer)

        nav = QHBoxLayout()
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.valueChanged.connect(self.change_image)
        self.file_label = QLabel("0 / 0")
        nav.addWidget(self.image_slider)
        nav.addWidget(self.file_label)
        vg.addLayout(nav)
        main.addWidget(viewer_group, 1)

        params = QGroupBox("グリーンバック検出・塗り替え")
        grid = QGridLayout(params)

        self.sample_label = QLabel()
        self.target_label = QLabel()
        self.sample_edit = QLineEdit()
        self.target_edit = QLineEdit()
        self.sample_edit.setPlaceholderText("例: 21,146,52")
        self.target_edit.setPlaceholderText("例: 0,0,0")
        pick_btn = QPushButton("スポイト: 画像をクリック")
        pick_btn.clicked.connect(lambda: self.set_status("プレビュー画像をクリックすると基準RGBを取得します"))

        grid.addWidget(QLabel("基準RGB:"), 0, 0)
        grid.addWidget(self.sample_edit, 0, 1)
        grid.addWidget(self.sample_label, 0, 2)
        grid.addWidget(pick_btn, 0, 3)

        grid.addWidget(QLabel("置換RGB:"), 1, 0)
        grid.addWidget(self.target_edit, 1, 1)
        grid.addWidget(self.target_label, 1, 2)

        self.hue_spin = QDoubleSpinBox()
        self.hue_spin.setRange(0.1, 90)
        self.hue_spin.setSingleStep(1)
        self.hue_spin.setSuffix(" °")
        self.sat_spin = QSpinBox()
        self.sat_spin.setRange(0, 255)
        self.value_min_spin = QSpinBox()
        self.value_min_spin.setRange(0, 255)
        self.value_max_spin = QSpinBox()
        self.value_max_spin.setRange(0, 255)
        self.dist_spin = QDoubleSpinBox()
        self.dist_spin.setRange(1, 441.7)
        self.dist_spin.setSingleStep(5)
        self.soft_spin = QDoubleSpinBox()
        self.soft_spin.setRange(0, 100)
        self.soft_spin.setSingleStep(1)
        self.area_spin = QSpinBox()
        self.area_spin.setRange(0, 10000000)

        rows = [
            ("色相の許容範囲:", self.hue_spin),
            ("最低彩度:", self.sat_spin),
            ("最低明度:", self.value_min_spin),
            ("最高明度:", self.value_max_spin),
            ("RGB距離:", self.dist_spin),
            ("境界の柔らかさ:", self.soft_spin),
            ("小領域除去(px):", self.area_spin),
        ]
        for r, (name, widget) in enumerate(rows, start=2):
            grid.addWidget(QLabel(name), r, 0)
            grid.addWidget(widget, r, 1)

        self.luma_check = QCheckBox("置換色の明るさを元画像に合わせる")
        self.rgb_check = QCheckBox("RGB距離も検出条件にする")
        grid.addWidget(self.luma_check, 2, 2, 1, 2)
        grid.addWidget(self.rgb_check, 3, 2, 1, 2)

        hint = QLabel(
            "HSVの色相・彩度・明度を基本条件として検出します。"
            " RGB距離は必要な場合だけ有効にしてください。"
        )
        hint.setWordWrap(True)
        grid.addWidget(hint, 3, 2, 4, 2)

        main.addWidget(params)

        buttons = QHBoxLayout()
        self.run_btn = QPushButton("すべての画像を処理")
        self.run_btn.clicked.connect(self.process_all)
        self.cancel_btn = QPushButton("処理を中止")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        self.save_settings_btn = QPushButton("設定を保存")
        self.save_settings_btn.clicked.connect(self.save_current_settings)
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.save_settings_btn)
        self.status = QLabel("")
        buttons.addWidget(self.status, 1)
        main.addLayout(buttons)

        for w in [self.sample_edit, self.target_edit, self.hue_spin, self.sat_spin,
                  self.value_min_spin, self.value_max_spin, self.dist_spin,
                  self.soft_spin, self.area_spin, self.luma_check, self.rgb_check]:
            if isinstance(w, QCheckBox):
                w.stateChanged.connect(self.schedule_preview)
            else:
                w.textChanged.connect(self.schedule_preview) if isinstance(w, QLineEdit) else w.valueChanged.connect(self.schedule_preview)

    def load_settings_to_ui(self):
        s = self.settings
        self.input_edit.setText(s["input_dir"])
        self.output_edit.setText(s["output_dir"])
        self.sample_edit.setText(",".join(map(str, s["sample_rgb"])))
        self.target_edit.setText(",".join(map(str, s["target_rgb"])))
        self.hue_spin.setValue(s["hue_range"])
        self.sat_spin.setValue(s["sat_min"])
        self.value_min_spin.setValue(s["value_min"])
        self.value_max_spin.setValue(s["value_max"])
        self.dist_spin.setValue(s["color_distance"])
        self.soft_spin.setValue(s["softness"])
        self.area_spin.setValue(s["min_area"])
        self.luma_check.setChecked(s["preserve_luminance"])
        self.rgb_check.setChecked(s.get("use_rgb_distance", False))
        self.update_color_labels()
        if self.input_edit.text() and Path(self.input_edit.text()).is_dir():
            self.scan_files()

    def parse_rgb(self, text, fallback):
        try:
            vals = [int(x.strip()) for x in text.split(",")]
            if len(vals) != 3 or any(v < 0 or v > 255 for v in vals):
                raise ValueError
            return vals
        except Exception:
            return fallback

    def get_params(self):
        sample = self.parse_rgb(self.sample_edit.text(), [21, 146, 52])
        target = self.parse_rgb(self.target_edit.text(), [0, 0, 0])
        return {
            "sample_rgb": sample,
            "target_rgb": target,
            "hue_range": self.hue_spin.value(),
            "sat_min": self.sat_spin.value(),
            "value_min": self.value_min_spin.value(),
            "value_max": self.value_max_spin.value(),
            "color_distance": self.dist_spin.value(),
            "softness": self.soft_spin.value(),
            "min_area": self.area_spin.value(),
            "preserve_luminance": self.luma_check.isChecked(),
            "use_rgb_distance": self.rgb_check.isChecked(),
        }

    def save_current_settings(self):
        self.settings.update(self.get_params())
        self.settings["input_dir"] = self.input_edit.text()
        self.settings["output_dir"] = self.output_edit.text()
        save_settings(self.settings)
        self.set_status("設定を保存しました")

    def closeEvent(self, event):
        self.save_current_settings()
        event.accept()

    def update_color_labels(self):
        sample = self.parse_rgb(self.sample_edit.text(), [21, 146, 52])
        target = self.parse_rgb(self.target_edit.text(), [0, 0, 0])
        self.sample_label.setStyleSheet(f"background:rgb({sample[0]},{sample[1]},{sample[2]}); min-width:50px;")
        self.target_label.setStyleSheet(f"background:rgb({target[0]},{target[1]},{target[2]}); min-width:50px;")

    def set_status(self, text):
        self.status.setText(text)

    def schedule_preview(self, *_):
        self.update_color_labels()
        self.preview_timer.start(120)

    def select_input(self):
        d = QFileDialog.getExistingDirectory(self, "入力フォルダ")
        if d:
            self.input_edit.setText(d)
            self.scan_files()

    def select_output(self):
        d = QFileDialog.getExistingDirectory(self, "出力フォルダ")
        if d:
            self.output_edit.setText(d)

    def scan_files(self):
        p = Path(self.input_edit.text())
        self.files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS])
        self.image_slider.blockSignals(True)
        self.image_slider.setRange(0, max(0, len(self.files) - 1))
        self.image_slider.setValue(0 if self.files else 0)
        self.image_slider.blockSignals(False)
        self.current_index = 0
        self.file_label.setText(f"{1 if self.files else 0} / {len(self.files)}")
        self.load_current_image()

    def change_image(self, value):
        self.current_index = value
        self.file_label.setText(f"{value + 1} / {len(self.files)}")
        self.load_current_image()

    def load_current_image(self):
        if not self.files:
            self.original_bgr = None
            self.viewer.set_image(None)
            return
        img = cv2.imread(str(self.files[self.current_index]), cv2.IMREAD_COLOR)
        if img is None:
            self.set_status(f"読み込み失敗: {self.files[self.current_index].name}")
            return
        self.original_bgr = img
        self.update_preview()

    def pick_color(self, r, g, b):
        self.sample_edit.setText(f"{r},{g},{b}")
        self.set_status(f"スポイト取得: RGB({r}, {g}, {b})")
        self.update_preview()

    def update_preview(self):
        if self.original_bgr is None:
            return
        p = self.get_params()

        # Large photos can make GUI sliders feel sluggish. Preview is resized
        # only for display; the actual batch processing uses the original size.
        h, w = self.original_bgr.shape[:2]
        max_side = 1400
        scale = min(1.0, max_side / max(h, w))
        if scale < 1:
            preview = cv2.resize(self.original_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            preview = self.original_bgr

        result_bgr, _ = make_preview(
            preview, p["sample_rgb"], p["target_rgb"], p["hue_range"],
            p["sat_min"], p["value_min"], p["value_max"], p["color_distance"],
            p["softness"], p["min_area"], p["preserve_luminance"], p["use_rgb_distance"]
        )
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        self.viewer.set_image(result_rgb)

    def cancel_processing(self):
        self.cancel_requested = True
        self.cancel_btn.setEnabled(False)
        self.set_status("中止要求を受け付けました。現在の画像の処理後に停止します...")
        QApplication.processEvents()

    def process_all(self):
        if not self.files:
            self.scan_files()
        if not self.files:
            QMessageBox.warning(self, "エラー", "入力フォルダに画像がありません。")
            return

        out_dir = Path(self.output_edit.text())
        if not self.output_edit.text().strip():
            QMessageBox.warning(self, "エラー", "出力フォルダを指定してください。")
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        p = self.get_params()
        self.save_current_settings()
        self.cancel_requested = False
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.save_settings_btn.setEnabled(False)
        QApplication.processEvents()

        ok = 0
        cancelled = False
        try:
            for i, src in enumerate(self.files):
                if self.cancel_requested:
                    cancelled = True
                    break
                img = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                result_bgr, _ = make_preview(
                    img, p["sample_rgb"], p["target_rgb"], p["hue_range"],
                    p["sat_min"], p["value_min"], p["value_max"], p["color_distance"],
                    p["softness"], p["min_area"], p["preserve_luminance"], p["use_rgb_distance"]
                )
                if self.cancel_requested:
                    cancelled = True
                    break
                dst = out_dir / src.name
                if cv2.imwrite(str(dst), result_bgr):
                    ok += 1
                self.status.setText(f"処理中... {i + 1}/{len(self.files)}")
                QApplication.processEvents()
        finally:
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.save_settings_btn.setEnabled(True)
            self.cancel_requested = False

        if cancelled:
            self.set_status(f"中止しました: {ok}/{len(self.files)} 枚を保存済み")
            QMessageBox.information(self, "中止", f"処理を中止しました。\n\n{ok} 枚を保存済みです。\n{out_dir}")
        else:
            self.set_status(f"完了: {ok}/{len(self.files)} 枚を保存しました")
            QMessageBox.information(self, "完了", f"{ok} 枚の画像を保存しました。\n\n{out_dir}")



def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Green Screen Recolor")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
