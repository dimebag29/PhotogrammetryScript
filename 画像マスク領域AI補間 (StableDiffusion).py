import os
import glob
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageOps

input_dir = r""

mask_dir = input_dir
output_dir = "./output_images"
os.makedirs(output_dir, exist_ok=True)

# SD Inpaintingモデルのロード
pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

# AIでの処理上限サイズ（大きすぎるとエラーになるため。VRAMに余裕があれば1024などを指定）
MAX_SD_SIZE = 1024

prompt = "seamless background, empty scene, high quality photograph, realistic texture, highly detailed, natural lighting"
negative_prompt = "people, person, human, body, crowd, blurry, distorted, unnatural, bad quality, artifacts"

# jpg, jpeg, png の各拡張子を取得
image_paths = []
for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
    image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

# リストの重複を排除
image_paths = list(set(image_paths))

# .mask.png を除外
image_paths = [
    path for path in image_paths
    if not path.lower().endswith(".mask.png")
]

for img_path in image_paths:
    base_name = os.path.basename(img_path)
    name_without_ext = os.path.splitext(base_name)[0]
    mask_path = os.path.join(mask_dir, f"{name_without_ext}.mask.png")
    
    if not os.path.exists(mask_path):
        print(f"Mask not found for {base_name}, skipping.")
        continue

    # 画像の読み込み
    init_image = Image.open(img_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("L")
    
    # マスクの反転（変更したい人物領域を白にする）
    inverted_mask = ImageOps.invert(mask_image)
    
    # --- 【改良1】サイズの計算 ---
    original_w, original_h = init_image.size
    
    # AIで処理可能なサイズに縮小するためのスケールを計算（長辺がMAX_SD_SIZEになるように）
    scale = min(MAX_SD_SIZE / original_w, MAX_SD_SIZE / original_h, 1.0)
    
    # Stable Diffusionは縦横が8の倍数である必要がある
    target_w = int((original_w * scale) // 8) * 8
    target_h = int((original_h * scale) // 8) * 8
    
    # AIに渡すためのリサイズ
    sd_image = init_image.resize((target_w, target_h), Image.LANCZOS)
    sd_mask = inverted_mask.resize((target_w, target_h), Image.LANCZOS)
    
    print(f"Processing {base_name} : Original {original_w}x{original_h} -> SD processing {target_w}x{target_h}")

    # 推論実行 (widthとheightを明示的に指定してアスペクト比を固定)
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=sd_image,
        mask_image=sd_mask,
        width=target_w,
        height=target_h,
        num_inference_steps=30,
        strength=0.99 
    ).images[0]
    
    # --- 【改良2】元のサイズに復元 ---
    result_resized = result.resize((original_w, original_h), Image.LANCZOS)
    
    # --- 【改良3】元画像との合成（コンポジット） ---
    # 白(255)の部分はAI生成画像を、黒(0)の部分は元画像を採用します
    # これにより、人物がいなかった領域の画質が100%維持されます
    final_image = Image.composite(result_resized, init_image, inverted_mask)
    
    # 保存
    output_path = os.path.join(output_dir, base_name)
    final_image.save(output_path)
    print(f"Saved: {base_name} (Final size: {original_w}x{original_h})\n")
