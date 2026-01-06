import cv2
import os

# 入力動画と出力ディレクトリ
VIDEO_PATH = r"4"
OUTPUT_DIR = "Export"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("動画を開けませんでした")

frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ゼロ埋め連番で保存
    filename = f"frame-{frame_index:06d}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)

    cv2.imwrite(output_path, frame)
    frame_index += 1

cap.release()

print(f"保存完了: {frame_index} フレーム")
