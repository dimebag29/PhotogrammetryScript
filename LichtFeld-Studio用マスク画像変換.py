import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

# 入力フォルダの絶対パス
INPUT_FOLDER = r""

PARENT_FOLDER = os.path.dirname(INPUT_FOLDER)
OUTPUT_FOLDER = os.path.join(PARENT_FOLDER, "masks")

IMAGE_EXTENSIONS = {".png", ".tga", ".bmp", ".tif", ".tiff", ".webp"}

# アルファ値→マスク値変換テーブル
LUT = [0 if a <= 127 else 255 for a in range(256)]


def process_single_image(filename):
    """1枚の画像を処理する関数"""
    input_path = os.path.join(INPUT_FOLDER, filename)
    base_name = os.path.splitext(filename)[0]

    try:
        with Image.open(input_path) as img:
            img = img.convert("RGBA")
            alpha = img.getchannel("A")
            mask = alpha.point(LUT)
            result = Image.merge("RGBA", (mask, mask, mask, mask))

            # masks フォルダに「ファイル名.png」として保存
            output_path = os.path.join(OUTPUT_FOLDER, base_name + ".png")
            result.save(output_path)

        return True, filename
    except Exception as e:
        return False, f"{filename} (Error: {e})"


def main():
    # 自動判定された OUTPUT_FOLDER (masks) がなければ作成
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 処理対象のファイルをリストアップ
    tasks = []
    if os.path.exists(INPUT_FOLDER):
        for filename in os.listdir(INPUT_FOLDER):
            input_path = os.path.join(INPUT_FOLDER, filename)

            if not os.path.isfile(input_path):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            tasks.append(filename)
    else:
        print(f"Error: INPUT_FOLDER が見つかりません: {INPUT_FOLDER}")
        return

    total_files = len(tasks)
    if total_files == 0:
        print(f"No images found in: {INPUT_FOLDER}")
        return

    print(f"Target: {INPUT_FOLDER}")
    print(f"Output: {OUTPUT_FOLDER}")
    print(f"Found {total_files} images. Starting parallel processing...\n")

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_single_image, filename): filename
            for filename in tasks
        }

        completed_count = 0

        for future in as_completed(futures):
            completed_count += 1
            success, message = future.result()

            progress_percent = (completed_count / total_files) * 100

            if success:
                print(
                    f"[{progress_percent:6.2f}%] ({completed_count}/{total_files}) Processed: {message}"
                )
            else:
                print(
                    f"[{progress_percent:6.2f}%] ({completed_count}/{total_files}) Failed: {message}"
                )

    print("\nDone!")


if __name__ == "__main__":
    main()
