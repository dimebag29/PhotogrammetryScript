import os
import numpy as np
from PIL import Image

def batch_color_replacement(folder_path):
    # 設定値
    TARGET_COLOR = np.array([67, 148, 240])
    # 255に対する17.5%の許容誤差
    THRESHOLD = 255 * 0.175

    # フォルダ内のファイルを確認
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 画像を開いてRGBに変換
                with Image.open(file_path) as img:
                    img_format = img.format # 元のフォーマットを保持
                    # アルファチャンネルがある場合も考慮してRGBAで読み込み、RGB部分で判定
                    img_array = np.array(img.convert('RGB'))
                    
                    # 各ピクセルと基準色の差分（絶対値）を計算
                    # axis=-1 でRGBチャネル方向の平均を取る
                    diff = np.abs(img_array - TARGET_COLOR)
                    avg_error = np.mean(diff, axis=-1)
                    
                    # 平均誤差が閾値以内のピクセルを特定するマスク
                    mask = avg_error <= THRESHOLD
                    
                    # マスクされた箇所を基準色で塗りつぶし
                    img_array[mask] = TARGET_COLOR
                    
                    # 配列を画像に戻して保存
                    result_img = Image.fromarray(img_array.astype(np.uint8))
                    result_img.save(file_path, format=img_format)
                    
                print(f"Processed: {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # 画像が入っているフォルダのパスを指定してください
    target_folder = r"" 
    
    if os.path.exists(target_folder):
        batch_color_replacement(target_folder)
        print("完了しました。")
    else:
        print(f"エラー: フォルダ '{target_folder}' が見つかりません。")
