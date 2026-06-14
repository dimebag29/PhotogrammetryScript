import os
from PIL import Image

INPUT_FOLDER = r""

OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "masks")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".tga", ".bmp", ".tif", ".tiff", ".webp"}

# アルファ値→マスク値変換テーブル
LUT = [0 if a <= 127 else 255 for a in range(256)]

for filename in os.listdir(INPUT_FOLDER):
    input_path = os.path.join(INPUT_FOLDER, filename)

    if not os.path.isfile(input_path):
        continue

    ext = os.path.splitext(filename)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        continue

    try:
        img = Image.open(input_path).convert("RGBA")

        # アルファチャンネル取得
        alpha = img.getchannel("A")

        # 0/255マスク化
        mask = alpha.point(LUT)

        # RGBA出力
        result = Image.merge("RGBA", (mask, mask, mask, mask))

        output_path = os.path.join(OUTPUT_FOLDER, filename + ".png")
        result.save(output_path)

        print(f"Processed: {filename}")

    except Exception as e:
        print(f"Error: {filename}")
        print(e)

print("Done")
