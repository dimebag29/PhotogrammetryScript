import subprocess
import os
import winsound
import time
from tqdm import tqdm

# 複数の動画ファイルをリストに入れる
fullpaths = [
    r""
]

# 視点リスト（例：12方向）
transforms = (
    (0, 0, 0),
    (30, 0, 0),
    (60, 0, 0),
    (90, 0, 0),
    (120, 0, 0),
    (150, 0, 0),
    (180, 0, 0),
    (-150, 0, 0),
    (-120, 0, 0),
    (-90, 0, 0),
    (-60, 0, 0),
    (-30, 0, 0),
)

fps = 2
gamma = 1.0

winsound.PlaySound("MailBeep", winsound.SND_ALIAS)

# 全体の処理ステップ数（ファイル数 × 視点数）
total_steps = len(fullpaths) * len(transforms)

# tqdm で進捗バーを表示
with tqdm(total=total_steps, desc="Processing", unit="task") as pbar:
    for fullpath in fullpaths:
        path = os.path.dirname(fullpath) + "\\"
        filiname, extension = os.path.splitext(os.path.basename(fullpath))
        extension = extension.lstrip('.')

        # 出力フォルダ
        output_path = fr'{path}Export' + '\\'
        os.makedirs(output_path, exist_ok=True)

        input_file_path = f'{path}{filiname}.{extension}'

        for index, transform in enumerate(transforms):
            # 出力ファイル名
            output_file_path = fr'{output_path}{filiname}_%04d_{index}.png'

            # v360オプション
            v360_options = ':'.join([
                'input=e',
                'output=rectilinear',
                'h_fov=121.28',
                'v_fov=90',
                'w=3413',
                'h=1920',
                f'yaw={transform[0]}',
                f'pitch={transform[1]}',
                f'roll={transform[2]}'
            ])

            # コマンド
            command = f'ffmpeg -y -i "{input_file_path}" -vf "v360={v360_options}, eq=gamma={gamma}" -qmin 1 -q 1 -r {fps} "{output_file_path}"'

            # ffmpegの出力を非表示にして実行
            subprocess.call(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # 進捗バー更新
            pbar.update(1)

winsound.PlaySound("MailBeep", winsound.SND_ALIAS)
time.sleep(1)
winsound.PlaySound("MailBeep", winsound.SND_ALIAS)

