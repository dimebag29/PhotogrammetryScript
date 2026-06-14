import os
import numpy as np
from PIL import Image
import cv2
from simple_lama_inpainting import SimpleLama

def batch_inpaint(target_dir, black_threshold=50, dilate_pixels=5):
    """
    black_threshold: 黒塗りと判定するRGBの最大値。JPGのノイズ対策で高めに設定（0-255）
    dilate_pixels: マスクを太らせるピクセル数。フチに残るグレーのノイズを完全に消し去ります。
    """
    # 1. 出力用フォルダの作成
    output_dir = os.path.join(target_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 2. SimpleLamaの初期化
    print("LaMaモデルを読み込んでいます...")
    lama = SimpleLama()
    
    # 3. 対応する画像拡張子
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    files = [f for f in os.listdir(target_dir) if f.lower().endswith(valid_extensions)]
    
    if not files:
        print(f"指定されたフォルダに画像が見つかりませんでした: {target_dir}")
        return

    print(f"処理を開始します。対象ファイル数: {len(files)}")

    for file_name in files:
        file_path = os.path.join(target_dir, file_name)
        print(f"処理中: {file_name} ...")
        
        try:
            # 4. 画像の読み込みとRGB変換
            img = Image.open(file_path)
            if img.mode == "RGBA":
                rgb_img = img.convert("RGB")
                alpha = np.array(img.split()[-1])
                has_alpha = True
            else:
                rgb_img = img.convert("RGB")
                has_alpha = False
                
            img_np = np.array(rgb_img)
            
            # 5. 【改良】しきい値を高めにして、黒〜濃いグレーの領域を大まかに抽出
            black_mask = (
                (img_np[:, :, 0] <= black_threshold) & 
                (img_np[:, :, 1] <= black_threshold) & 
                (img_np[:, :, 2] <= black_threshold)
            )
            
            if has_alpha:
                mask_condition = black_mask | (alpha == 0)
            else:
                mask_condition = black_mask
            
            # 一旦マスクを生成 (0 or 255)
            mask_array = np.where(mask_condition, 255, 0).astype(np.uint8)
            
            # マスク領域が全くない場合はスキップ
            if mask_array.max() == 0:
                print(f" -> スキップ: マスク領域（黒塗り）が検出されませんでした。")
                continue
            
            # 6. 【最重要追加】マスクの膨張処理（Dilation）
            # 人物の輪郭部分に残る微妙に明るい中間色のグレーを、周囲ごと巻き込んで消し去ります
            if dilate_pixels > 0:
                kernel = np.ones((dilate_pixels, dilate_pixels), np.uint8)
                mask_array = cv2.dilate(mask_array, kernel, iterations=1)
            
            # SimpleLama用にPILイメージに変換
            mask = Image.fromarray(mask_array).convert("L")
                
            # 7. AIインペインティング実行
            result = lama(rgb_img, mask)
            
            # 8. 画像の保存
            base_name, _ = os.path.splitext(file_name)
            output_path = os.path.join(output_dir, f"{base_name}_inpainted.png")
            result.save(output_path)
            print(f" -> 保存完了: {output_path}")
            
        except Exception as e:
            print(f" -> エラーが発生しました ({file_name}): {e}")

    print("\nすべての処理が完了しました！")

if __name__ == "__main__":
    # 対象のフォルダパス
    TARGET_DIRECTORY = r""
    
    # しきい値を50（かなり明るい黒まで許容）、膨張を5ピクセルに設定して確実に消し込みます
    batch_inpaint(TARGET_DIRECTORY, black_threshold=50, dilate_pixels=5)
