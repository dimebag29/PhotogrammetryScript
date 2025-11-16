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

# ADE20KクラスID 2:空, 12:人, 20:車 → (R, G, B, A)
fill_colors = {
     2: ( 67, 148, 240, 255),
    12: (  0,   0,   0,   0)
}

tile_size = 512  # SegFormerの学習解像度

# === モデル読み込み ===
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model.eval()

# CUDA設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

use_fp16 = False  # GTX1080では無効


# === マスク処理関数（RGBA対応） ===
def process_tile(tile_img):
    inputs = feature_extractor(images=tile_img, return_tensors="pt").to(device)
    if use_fp16 and device.type == "cuda":
        inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = logits.argmax(dim=1)[0].cpu().numpy()

    # SegFormer 出力マスク
    mask = Image.fromarray(pred.astype(np.uint8)).resize(tile_img.size, resample=Image.NEAREST)
    mask = np.array(mask)

    # タイル画像を RGBA に変換
    tile_np = np.array(tile_img.convert("RGBA"))

    # === 指定クラスを塗りつぶし（RGBA対応） ===
    for class_id, color_rgba in fill_colors.items():
        class_mask = (mask == class_id)

        # 1px 膨張
        class_mask_dilated = binary_dilation(class_mask, structure=np.ones((3, 3)))

        r, g, b, a = color_rgba
        tile_np[class_mask_dilated] = [r, g, b, a]

    return Image.fromarray(tile_np)


# === 入出力処理 ===
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

    # 出力画像を RGBA で作成
    result_img = Image.new("RGBA", (width, height))

    # === タイル処理（下から上へ）===

    # top のリストを作成して逆順に
    top_positions = list(range(0, height, tile_size))
    top_positions.reverse()  # 下 → 上

    for top in top_positions:
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            tile = image.crop((left, top, right, bottom))

            processed_tile = process_tile(tile)
            result_img.paste(processed_tile, (left, top))

    # PNGで RGBA 保存
    result_img.save(output_img_path, compress_level=1)

    processed_count += 1
    elapsed = time.time() - start_time
    avg_time_per_img = elapsed / processed_count
    remaining = (remaining_to_process - processed_count) * avg_time_per_img
    finish_time = datetime.now() + timedelta(seconds=remaining)

    print(f"[{idx}/{total_files}] {filename} 保存しました")
    print(f"  残り {remaining_to_process - processed_count} 枚")
    print(f"  終了予定時刻: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}\n")


