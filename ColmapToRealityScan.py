import os
import math
import shutil
from pathlib import Path

# ============================================================
# 設定
# ============================================================

# COLMAPで「Export model as text」したフォルダ
COLMAP_MODEL_DIR = r""

# COLMAPで使用した元画像フォルダ
IMAGE_DIR = r""



# XMPの出力先
# IMAGE_DIRと同じにすると、画像の横にXMPを作成します
OUTPUT_XMP_DIR = IMAGE_DIR

# RealityScanでのPose Prior
#
# "initial" : COLMAPの位置を初期値として使用し、RealityScanが調整可能
# "locked"  : COLMAPの位置・姿勢を固定
#
# COLMAPの位置推定結果をそのまま使いたいなら "locked"
POSE_PRIOR = "locked"

# RealityScanで再計算させるかどうか
IN_TEXTURING = 1
IN_MESHING = 1

# COLMAPの内部パラメータをXMPに書き込む
WRITE_INTRINSICS = True

# ============================================================
# COLMAP quaternion → rotation matrix
# COLMAP:
# q = [qw, qx, qy, qz]
# ============================================================

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    return [
        [
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ],
        [
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw),
        ],
        [
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy),
        ],
    ]


# ============================================================
# 行列の転置
# ============================================================

def transpose_matrix(m):
    return [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]


# ============================================================
# 行列 × ベクトル
# ============================================================

def mat_vec_mul(m, v):
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


# ============================================================
# 数値フォーマット
# ============================================================

def fmt(v):
    return f"{v:.15f}"


# ============================================================
# cameras.txt 読み込み
# ============================================================

def read_cameras(path):
    cameras = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]

            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }

    return cameras


# ============================================================
# COLMAPカメラ → RealityScan XMPパラメータ
# ============================================================

def camera_to_xmp(camera):
    model = camera["model"]
    width = camera["width"]
    height = camera["height"]
    p = camera["params"]

    fx = None
    fy = None
    cx = width / 2
    cy = height / 2

    # ----------------------------------------
    # SIMPLE_PINHOLE
    # f cx cy
    # ----------------------------------------
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        fx = fy = f

    # ----------------------------------------
    # PINHOLE
    # fx fy cx cy
    # ----------------------------------------
    elif model == "PINHOLE":
        fx, fy, cx, cy = p

    # ----------------------------------------
    # SIMPLE_RADIAL
    # f cx cy k
    # ----------------------------------------
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = p
        fx = fy = f

    # ----------------------------------------
    # RADIAL
    # f cx cy k1 k2
    # ----------------------------------------
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = p
        fx = fy = f

    # ----------------------------------------
    # OPENCV
    # fx fy cx cy k1 k2 p1 p2
    # ----------------------------------------
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p

    # ----------------------------------------
    # FULL_OPENCV
    # fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6
    # ----------------------------------------
    elif model == "FULL_OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = p

    # ----------------------------------------
    # OPENCV_FISHEYE
    # ----------------------------------------
    elif model == "OPENCV_FISHEYE":
        fx, fy, cx, cy, k1, k2, k3, k4 = p

    # ----------------------------------------
    # FOV
    # fx fy cx cy omega
    # ----------------------------------------
    elif model == "FOV":
        fx, fy, cx, cy, omega = p

    else:
        print(f"[WARNING] 未対応のカメラモデル: {model}")
        print("         焦点距離と主点を近似値で出力します")

        fx = fy = max(width, height)
        cx = width / 2
        cy = height / 2

    # ------------------------------------------------
    # RealityScanのFocalLength35mm
    #
    # RealityScan XMPでは35mm換算として扱われ、
    # 36mm幅を基準に
    #
    # f35 = fx / image_width * 36
    # ------------------------------------------------

    focal_length_35mm = fx / width * 36.0

    # ------------------------------------------------
    # PrincipalPointU/V
    #
    # XMPのPrincipalPointは画像中心からの正規化値
    # ------------------------------------------------

    principal_u = (cx - width / 2) / width
    principal_v = (cy - height / 2) / width

    # ------------------------------------------------
    # COLMAPの歪み係数
    # RealityScan brown3:
    #
    # k1 k2 k3 t1 t2
    #
    # として扱う
    # ------------------------------------------------

    distortion = [0.0] * 6

    if model == "SIMPLE_RADIAL":
        distortion[0] = k1

    elif model == "RADIAL":
        distortion[0] = k1
        distortion[1] = k2

    elif model == "OPENCV":
        distortion[0] = k1
        distortion[1] = k2
        distortion[2] = 0.0
        distortion[3] = p1
        distortion[4] = p2

    elif model == "FULL_OPENCV":
        distortion[0] = k1
        distortion[1] = k2
        distortion[2] = k3
        distortion[3] = p1
        distortion[4] = p2

    # OPENCV_FISHEYEはbrown3とはモデルが違うため
    # 安全のためゼロにする
    elif model == "OPENCV_FISHEYE":
        distortion = [0.0] * 6

    return {
        "focal_length_35mm": focal_length_35mm,
        "principal_u": principal_u,
        "principal_v": principal_v,
        "distortion": distortion,
        "width": width,
        "height": height,
    }


# ============================================================
# XMP生成
# ============================================================

def create_xmp(
    rotation,
    position,
    camera_info
):
    r = " ".join(fmt(x) for row in rotation for x in row)
    pos = " ".join(fmt(x) for x in position)

    focal = camera_info["focal_length_35mm"]
    pu = camera_info["principal_u"]
    pv = camera_info["principal_v"]

    distortion = " ".join(
        fmt(x) for x in camera_info["distortion"]
    )

    return f'''<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description
    xcr:Version="3"
    xcr:PosePrior="{POSE_PRIOR}"
    xcr:Coordinates="absolute"
    xcr:DistortionModel="brown3"
    xcr:FocalLength35mm="{fmt(focal)}"
    xcr:Skew="0"
    xcr:AspectRatio="1"
    xcr:PrincipalPointU="{fmt(pu)}"
    xcr:PrincipalPointV="{fmt(pv)}"
    xcr:CalibrationPrior="initial"
    xcr:CalibrationGroup="-1"
    xcr:DistortionGroup="-1"
    xcr:InTexturing="{IN_TEXTURING}"
    xcr:InMeshing="{IN_MESHING}"
    xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#">
<xcr:Rotation>{r}</xcr:Rotation>
<xcr:Position>{pos}</xcr:Position>
<xcr:DistortionCoeficients>{distortion}</xcr:DistortionCoeficients>
</rdf:Description>
</rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''


# ============================================================
# images.txt 読み込み
# ============================================================

def process_images(images_txt, cameras):
    count = 0
    skipped = 0

    with open(images_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()

        # COLMAP images.txt:
        #
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        #
        if len(parts) < 10:
            i += 1
            continue

        image_id = int(parts[0])

        qw = float(parts[1])
        qx = float(parts[2])
        qy = float(parts[3])
        qz = float(parts[4])

        tx = float(parts[5])
        ty = float(parts[6])
        tz = float(parts[7])

        camera_id = int(parts[8])

        image_name = " ".join(parts[9:])

        # 次の行は2D特徴点なのでスキップ
        i += 2

        if camera_id not in cameras:
            print(
                f"[WARNING] Camera ID {camera_id} "
                f"がcameras.txtにありません: {image_name}"
            )
            skipped += 1
            continue

        # ----------------------------------------
        # Quaternion → COLMAPのworld-to-camera R
        # ----------------------------------------

        R = quaternion_to_rotation_matrix(
            qw, qx, qy, qz
        )

        # ----------------------------------------
        # Camera center
        #
        # x_cam = R * x_world + t
        #
        # C = -R^T * t
        # ----------------------------------------

        Rt = transpose_matrix(R)

        C = mat_vec_mul(
            Rt,
            [-tx, -ty, -tz]
        )

        camera_info = camera_to_xmp(
            cameras[camera_id]
        )

        xmp = create_xmp(
            R,
            C,
            camera_info
        )

        # ----------------------------------------
        # 出力ファイル
        # ----------------------------------------

        image_path = Path(image_name)

        output_path = (
            Path(OUTPUT_XMP_DIR) /
            image_path.with_suffix(".xmp")
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8-sig",
            newline="\n"
        ) as out:
            out.write(xmp)

        count += 1

        if count % 100 == 0:
            print(
                f"{count} images processed..."
            )

    return count, skipped


# ============================================================
# メイン
# ============================================================

def main():
    model_dir = Path(COLMAP_MODEL_DIR)

    cameras_txt = model_dir / "cameras.txt"
    images_txt = model_dir / "images.txt"

    if not cameras_txt.exists():
        raise FileNotFoundError(
            f"cameras.txt がありません:\n{cameras_txt}\n\n"
            "COLMAPで Export model as text を実行してください。"
        )

    if not images_txt.exists():
        raise FileNotFoundError(
            f"images.txt がありません:\n{images_txt}\n\n"
            "COLMAPで Export model as text を実行してください。"
        )

    print("==========================================")
    print(" COLMAP → RealityScan XMP Converter")
    print("==========================================")
    print()
    print(f"COLMAP model : {model_dir}")
    print(f"Image folder : {IMAGE_DIR}")
    print(f"XMP output   : {OUTPUT_XMP_DIR}")
    print(f"Pose prior   : {POSE_PRIOR}")
    print()

    cameras = read_cameras(cameras_txt)

    print(
        f"COLMAP cameras: {len(cameras)}"
    )

    count, skipped = process_images(
        images_txt,
        cameras
    )

    print()
    print("==========================================")
    print("完了")
    print("==========================================")
    print(f"XMP generated : {count}")
    print(f"Skipped       : {skipped}")
    print()

    if OUTPUT_XMP_DIR == IMAGE_DIR:
        print(
            "画像フォルダにXMPを直接作成しました。"
        )
    else:
        print(
            "XMPを画像フォルダへコピーしてから"
            "RealityScanへ読み込んでください。"
        )


if __name__ == "__main__":
    main()
