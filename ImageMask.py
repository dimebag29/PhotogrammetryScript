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

# 画像が入っているフォルダと出力フォルダ指定
input_path = r""
output_path = r""
os.makedirs(output_path, exist_ok=True)

# ADE20KにおけるクラスID（複数指定可能）2:空 12:人 20:車
mask_class_ids = [2, 12, 20]

# 画像をこの解像度で切り出して、それぞれマスク処理を行う。(ADE20Kの学習モデルが512x512なのでこれに合わせる)
tile_size = 512

# モデルと前処理の読み込み
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model.eval()

# CUDAデバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# 半精度推論を有効化（VRAM削減＆高速化）GTX1080はTensorCoreがなく使えないのでFalse
use_fp16 = False
if use_fp16 and device.type == "cuda":
    model.half()

# 透過処理
def process_tile(tile_img):
    # 画像を前処理してGPUに送る
    inputs = feature_extractor(images=tile_img, return_tensors="pt").to(device)
    if use_fp16 and device.type == "cuda":
        inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # GPU上でargmax → CPUに戻す
        pred = logits.argmax(dim=1)[0].cpu().numpy()

    # マスク生成
    mask = Image.fromarray(pred.astype(np.uint8)).resize(tile_img.size, resample=Image.NEAREST)
    mask = np.array(mask)
    any_mask = np.isin(mask, mask_class_ids)

    # 1px膨張処理
    any_mask_dilated = binary_dilation(any_mask, structure=np.ones((3, 3)))

    # αチャンネル適用
    alpha = np.where(any_mask_dilated, 0, 255).astype(np.uint8)
    tile_rgba = tile_img.convert("RGBA")
    tile_np = np.array(tile_rgba)
    tile_np[..., 3] = alpha
    return Image.fromarray(tile_np)

# ファイル一覧取得
file_list = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
total_files = len(file_list)

start_time = None
processed_count = 0  # 実際に処理した枚数
remaining_to_process = sum(1 for f in file_list if not os.path.exists(os.path.join(output_path, os.path.splitext(f)[0] + ".png")))

# 画像を処理
for idx, filename in enumerate(file_list, start=1):
    output_file = os.path.splitext(filename)[0] + ".png"
    output_img_path = os.path.join(output_path, output_file)

    # 処理済みならスキップ
    if os.path.exists(output_img_path):
        print(f"[{idx}/{total_files}] {filename} → 既に存在するためスキップ")
        continue

    if start_time is None:
        start_time = time.time()  # 最初の処理開始時刻を記録

    img_path = os.path.join(input_path, filename)
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    result_img = Image.new("RGBA", (width, height))

    # タイルごとに処理
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            tile = image.crop((left, top, right, bottom))
            processed_tile = process_tile(tile)
            result_img.paste(processed_tile, (left, top))

    # 保存
    result_img.save(output_img_path)

    # 経過時間計算
    processed_count += 1
    elapsed = time.time() - start_time
    avg_time_per_img = elapsed / processed_count
    remaining = (remaining_to_process - processed_count) * avg_time_per_img
    finish_time = datetime.now() + timedelta(seconds=remaining)

    print(f"[{idx}/{total_files}] {filename} 保存しました")
    print(f"  残り {remaining_to_process - processed_count} 枚")
    print(f"  終了予定時刻: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
