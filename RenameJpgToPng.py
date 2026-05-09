import os

# 対象フォルダ
folder_path = r""

# フォルダ内のファイルを取得
for filename in os.listdir(folder_path):

    # .jpg のみ対象
    if filename.lower().endswith(".jpg"):

        old_path = os.path.join(folder_path, filename)

        # 拡張子だけ .png に変更
        new_filename = os.path.splitext(filename)[0] + ".png"
        new_path = os.path.join(folder_path, new_filename)

        # リネーム実行
        os.rename(old_path, new_path)

        print(f"Renamed: {filename} -> {new_filename}")

print("完了")
