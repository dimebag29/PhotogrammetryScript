import cv2
import os
import time
from natsort import natsorted  # pip install natsort

# === 設定 ===
input_folder = r""
fps = 29.970030  # 出力動画のフレームレート

# === 画像ファイル一覧 ===
images = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
images = natsorted(images)
total_frames = len(images)

if not images:
    raise ValueError("画像が見つかりません")

# 最初の画像からサイズを取得
first_img = cv2.imread(os.path.join(input_folder, images[0]))
height, width, _ = first_img.shape

# 画像を等間隔で割り当て
frame_indices = [int(i * len(images) / total_frames) for i in range(total_frames)]

# === 全画像をメモリにキャッシュ（リサイズ済み） ===
print("画像を読み込み中...")
cache = []
for i, img_name in enumerate(images):
    img = cv2.imread(os.path.join(input_folder, img_name))
    if img is None:
        continue
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))
    cache.append(img)
    
    print(f"画像読み込み完了: {i+1} / {total_frames} 枚")

# === 動画作成 ===
fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter(input_folder + ".mp4", fourcc, fps, (width, height))

print("動画作成を開始します...")
start_time = time.time()

for i, idx in enumerate(frame_indices, start=1):
    out.write(cache[min(idx, len(cache)-1)])

    # --- 進捗表示 ---
    if i % 500 == 0 or i == total_frames:  # 500フレームごとに表示
        elapsed = time.time() - start_time
        progress = i / total_frames
        remaining = elapsed / progress - elapsed
        print(f"[{progress*100:6.2f}%] 完了, 残り時間: {remaining/60:.2f} 分")

out.release()
print("動画を書き出しました:")
