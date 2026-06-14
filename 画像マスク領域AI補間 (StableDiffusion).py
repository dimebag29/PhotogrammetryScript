import os
import glob
from PIL import Image, ImageFilter, ImageOps
import torch
# SDXLベースのモデルIDに対して、正しいPipelineクラスを使用
from diffusers import StableDiffusionXLInpaintPipeline
from tqdm import tqdm

# ==========================================
# 設定（人物の影を消し去るための最適化）
# ==========================================
# 処理対象の画像が入っているフォルダのパス
TARGET_DIR = r""

# 結果を保存するフォルダ名（TARGET_DIR 内に作成されます）
OUTPUT_DIR_NAME = "Stable_Diffusion_Deep_Clean_Results"

# SDXLベースの高性能インペイントモデル（認証不要）
# このモデルは、StableDiffusionXLInpaintPipeline で動かす必要があります。
MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

# --- 【超重要】影（ゴースト）撲滅設定 ---
# 影やブラーを完全に覆い隠すため、マスクを外側に大きく拡張します（デフォルト2 -> 15～30を推奨）
# 影がまだ残る場合は、この数字をさらに大きくしてください。
MASK_DILATE_PX = 15

# 拡張したマスクの境界を滑らかにして、周囲と馴染ませるためのぼかし量（0以上、MASK_DILATE_PXの半分程度を推奨）
MASK_BLUR_PX = 10

# AIが描き直す強度。0.0～1.0（1.0で完全に新しい背景を描く。通常は1.0でOK）
STRENGTH = 1.0

# 描画のパラメータ（「影」や「幽霊」を出すなと強く指示）
# clean background だけでは影を「テクスチャ」と勘違いすることがあるため、プロンプトを具体化
PROMPT = "perfect flawless clear ground pavement without any shadow, clean empty street, high resolution"
#PROMPT = "clean background, flawless background, seamless texture, highly detailed"

# NEGATIVEに影、人物の残像、ブラー、ゴーストなどを追加
#NEGATIVE_PROMPT = "people, person, human, character, shadow, cast shadow, ghost, motion blur, blurry, artifact, bad anatomy, footprints"
NEGATIVE_PROMPT = "people, person, human, character, blurry, bad anatomy"

# ステップ数を増やして、描き込みの精度を上げる（時間がかかりますが、きれいになります）
STEPS = 50 
# ==========================================


def extract_and_expand_mask_from_alpha(image_path, dilate_px=25, blur_px=10):
    """
    RGBA画像のアルファチャンネル（A=0）からマスク画像を生成し、
    影をカバーするために大幅に拡張し、かつ境界をぼかす関数。
    """
    orig_img = Image.open(image_path).convert("RGBA")
    alpha = orig_img.split()[-1]
    
    # A=0 の部分（完全に透明な部分＝人物がいるはずの場所）を白(255)、それ以外を黒(0)にするマスクを作成
    mask = alpha.point(lambda p: 255 if p == 0 else 0)
    
    # 【ゴースト対策1】マスクを外側に大きく拡張（Dilate）
    # 人物の周囲にある半透明の影やブラーを完全に覆い隠すため、大きなフィルタをかけます
    if dilate_px > 0:
        filter_size = dilate_px * 2 + 1 # 奇数である必要あり
        mask = mask.filter(ImageFilter.MaxFilter(size=filter_size))
    
    # 【ゴースト対策2】マスクの境界をぼかす（GaussianBlur）
    # 拡張したマスクがパキッとしすぎると、AIが描いた部分と元画像の境界が不自然になるため、
    # 境界を滑らかにグラデーションさせます
    if blur_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_px))
    
    return orig_img, mask


def main():
    # ローカルGPU (CUDA) の確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("[Warning] GPU (CUDA) が検出されません。CPUでのローカルAI処理は非常に低速になります。")

    output_dir = os.path.join(TARGET_DIR, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)

    print("ローカルAIモデルをVRAMに読み込んでいます（初回のみ自動ダウンロードが走ります）...")
    
    # 【修正点】SDXLベースのモデルなので、StableDiffusionXLInpaintPipeline を使用
    # これが前のコードのエラーの原因です。
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        # 最新のdiffusersではsafety_checkerの指定方法が異なる場合がありますが、
        # SDXLインペイントでは通常、safety_checkerはNoneまたは未指定で動きます。
    )
    pipe = pipe.to(device)
    
    # VRAMを節約して効率化
    if device == "cuda":
        pipe.enable_attention_slicing()
        # VRAMが少ない場合はこちらも有効に（少し遅くなります）
        # pipe.enable_model_cpu_offload()

    # 対応する画像拡張子
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.webp')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(TARGET_DIR, ext)))

    if not image_files:
        print(f"指定されたフォルダに画像が見つかりませんでした: {TARGET_DIR}")
        return

    print(f"合計 {len(image_files)} 件の画像を、影を含めて完全に補間します。")
    print(f"マスク拡張量: {MASK_DILATE_PX}px, ステップ数: {STEPS}, モデル: {MODEL_ID}")

    # 一括処理ループ
    for img_path in tqdm(image_files, desc="SDXL Inpaint Loop"):
        try:
            filename = os.path.basename(img_path)
            # PNGで保存することを推奨（アルファチャンネルを維持したいため）
            save_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")

            if os.path.exists(save_path):
                print(f" スキップ（存在します）: {filename}")
                continue

            # 元画像(RGBA)と、影までカバーするように拡張・平滑化したマスク(L)の取得
            orig_image, mask_image = extract_and_expand_mask_from_alpha(
                img_path, dilate_px=MASK_DILATE_PX, blur_px=MASK_BLUR_PX
            )
            orig_width, orig_height = orig_image.size

            # 透明領域の存在チェック
            if mask_image.getextrema() == (0, 0):
                print(f" スキップ（透明領域なし）: {filename}")
                continue

            # AIに入力するためにRGBに変換
            init_image_rgb = orig_image.convert("RGB")

            # 【SDXLゴースト対策】入力画像を8の倍数にリサイズ（SDXLの潜在空間仕様に合わせる）
            input_w = (orig_width // 8) * 8
            input_h = (orig_height // 8) * 8
            
            # リサイズが必要な場合
            if input_w != orig_width or input_h != orig_height:
                sdxl_input_image = init_image_rgb.resize((input_w, input_h), Image.Resampling.LANCZOS)
                # マスクは二値化（または滑らかなグラデーション）を維持するためNEARESTまたはBILINEAR
                sdxl_input_mask = mask_image.resize((input_w, input_h), Image.Resampling.BILINEAR)
            else:
                sdxl_input_image = init_image_rgb
                sdxl_input_mask = mask_image

            # ローカルGPUによるインペインティング実行
            with torch.inference_mode():
                ai_output = pipe(
                    prompt=PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=sdxl_input_image,
                    mask_image=sdxl_input_mask,
                    num_inference_steps=STEPS,
                    strength=STRENGTH,  # 強度を追加
                ).images[0]

            # 【解像度・サイズ完全維持】元のサイズに戻す
            ai_output_resized = ai_output.resize((orig_width, orig_height), Image.Resampling.LANCZOS)

            # 【マスク外完全無傷マージ】
            # AIが描いた画像（ai_output_resized）を、
            # 拡張・ぼかしを入れたマスク（mask_image）を使って、
            # 元画像のRGB（init_image_rgb）の上に合成します。
            final_rgb = Image.composite(ai_output_resized, init_image_rgb, mask_image)
            
            # 保存（品質を重視してPNG、または高品質JPEG）
            final_rgb.save(save_path, "PNG") # 拡張子に関わらずPNGで保存

        except Exception as e:
            print(f"\n[Error] ファイル {os.path.basename(img_path)} の処理中にエラーが発生しました: {e}")

    print(f"\nすべての処理が完了しました！ 人物のゴーストが消えているか確認してください。\n結果は以下に保存されています:\n{output_dir}")


if __name__ == "__main__":
    main()
