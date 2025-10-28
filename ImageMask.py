# GTX1080
# Python         3.12.0

# torch          2.7.1+cu118
# torchaudio     2.7.1+cu118
# torchvision    0.22.1+cu118

import os
import time
import torch
import numpy as np
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from scipy.ndimage import binary_dilation
from datetime import datetime, timedelta

# === 設定 ===
input_path = r""
output_path = r""
os.makedirs(output_path, exist_ok=True)

# ADE20KクラスID 2:空, 12:人, 20:車
mask_class_ids = [2, 12, 20]

# 塗りつぶし色 (R, G, B)
fill_colors = {
    2: (0, 255, 255),    # Cyan
    12: (255, 0, 255),   # Magenta
    20: (255, 255, 0),   # Yellow
}

tile_size = 512  # SegFormerモデルの学習解像度

# === モデル読み込み ===
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model.eval()

# CUDA設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

use_fp16 = False  # GTX1080では無効

# === マスク処理 ===
def process_tile(tile_img):
    inputs = feature_extractor(images=tile_img, return_tensors="pt").to(device)
    if use_fp16 and device.type == "cuda":
        inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = logits.argmax(dim=1)[0].cpu().numpy()

    mask = Image.fromarray(pred.astype(np.uint8)).resize(tile_img.size, resample=Image.NEAREST)
    mask = np.array(mask)

    # RGB画像をnumpy配列に
    tile_np = np.array(tile_img.convert("RGB"))

    # 各クラスごとに塗りつぶし処理
    for class_id, color in fill_colors.items():
        class_mask = (mask == class_id)
        # 1px膨張処理
        class_mask_dilated = binary_dilation(class_mask, structure=np.ones((3, 3)))
        tile_np[class_mask_dilated] = color

    return Image.fromarray(tile_np)

# === ファイル処理 ===
file_list = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
total_files = len(file_list)
start_time = None
processed_count = 0
remaining_to_process = sum(1 for f in file_list if not os.path.exists(os.path.join(output_path, os.path.splitext(f)[0] + ".png")))

for idx, filename in enumerate(file_list, start=1):
    output_file = os.path.splitext(filename)[0] + ".png"
    output_img_path = os.path.join(output_path, output_file)

    if os.path.exists(output_img_path):
        print(f"[{idx}/{total_files}] {filename} → 既に存在するためスキップ")
        continue

    if start_time is None:
        start_time = time.time()

    img_path = os.path.join(input_path, filename)
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    result_img = Image.new("RGB", (width, height))

    # タイルごとに処理
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            tile = image.crop((left, top, right, bottom))
            processed_tile = process_tile(tile)
            result_img.paste(processed_tile, (left, top))

    result_img.save(output_img_path)

    processed_count += 1
    elapsed = time.time() - start_time
    avg_time_per_img = elapsed / processed_count
    remaining = (remaining_to_process - processed_count) * avg_time_per_img
    finish_time = datetime.now() + timedelta(seconds=remaining)

    print(f"[{idx}/{total_files}] {filename} 保存しました")
    print(f"  残り {remaining_to_process - processed_count} 枚")
    print(f"  終了予定時刻: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
