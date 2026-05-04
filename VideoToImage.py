import tkinter as tk
from tkinter import filedialog, ttk
import subprocess
import os
import threading
import time
from datetime import datetime, timedelta
import glob

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MP4 → 画像抽出ツール")

        self.files = []
        self.stop_flag = False
        self.current_process = None
        self.total_expected = 0

        # ffmpegチェック
        self.ffmpeg_label = tk.Label(root, text=self.check_ffmpeg())
        self.ffmpeg_label.pack()

        # ファイル選択
        tk.Button(root, text="MP4追加", command=self.add_files).pack()
        self.file_list = tk.Listbox(root, width=60)
        self.file_list.pack()

        # 秒数間隔＋予想枚数（横並び）
        frame1 = tk.Frame(root)
        frame1.pack()

        tk.Label(frame1, text="秒数間隔:").pack(side=tk.LEFT)

        self.interval_entry = tk.Entry(frame1, width=10)
        self.interval_entry.pack(side=tk.LEFT)
        self.interval_entry.bind("<KeyRelease>", self.update_estimate)

        self.estimate_label = tk.Label(frame1, text="　予想枚数: -")
        self.estimate_label.pack(side=tk.LEFT)

        # 形式選択（横並び）
        frame2 = tk.Frame(root)
        frame2.pack()

        self.format_var = tk.StringVar(value="png")
        tk.Radiobutton(frame2, text="PNG", variable=self.format_var, value="png").pack(side=tk.LEFT)
        tk.Radiobutton(frame2, text="JPG", variable=self.format_var, value="jpg").pack(side=tk.LEFT)

        # プログレスバー
        self.progress = ttk.Progressbar(root, length=300)
        self.progress.pack()

        self.time_label = tk.Label(root, text="終了予定時間: -")
        self.time_label.pack()

        # 開始・停止（横並び）
        frame3 = tk.Frame(root)
        frame3.pack()

        tk.Button(frame3, text="開始", command=self.start).pack(side=tk.LEFT)
        tk.Button(frame3, text="停止", command=self.stop).pack(side=tk.LEFT)

    def check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return "ffmpeg: 利用可能"
        except:
            return "ffmpeg: 利用不可"

    def add_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("MP4 files", "*.mp4")])
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.file_list.insert(tk.END, p)

    def get_duration(self, file):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            return float(result.stdout.strip())
        except:
            return 0

    def update_estimate(self, event=None):
        try:
            interval = float(self.interval_entry.get())
            if interval <= 0:
                raise ValueError
        except:
            self.estimate_label.config(text="　予想枚数: -")
            self.total_expected = 0
            return

        total = 0
        for f in self.files:
            duration = self.get_duration(f)
            total += int(duration / interval)

        self.total_expected = total
        self.estimate_label.config(text=f"　予想枚数: {total}")

    def start(self):
        self.stop_flag = False
        thread = threading.Thread(target=self.process)
        thread.start()

    def stop(self):
        self.stop_flag = True
        if self.current_process:
            self.current_process.kill()

    def count_images(self):
        total = 0
        for f in self.files:
            base = os.path.splitext(os.path.basename(f))[0]
            out_dir = os.path.join(os.path.dirname(f), "Img")
            ext = self.format_var.get()
            pattern = os.path.join(out_dir, f"{base}_*.{ext}")
            total += len(glob.glob(pattern))
        return total

    def process(self):
        try:
            interval = float(self.interval_entry.get())
            if interval <= 0:
                return
        except:
            return

        start_time = time.time()

        for file in self.files:
            if self.stop_flag:
                break

            base = os.path.splitext(os.path.basename(file))[0]
            out_dir = os.path.join(os.path.dirname(file), "Img")
            os.makedirs(out_dir, exist_ok=True)

            ext = self.format_var.get()
            output_pattern = os.path.join(out_dir, f"{base}_%06d.{ext}")

            cmd = [
                "ffmpeg",
                "-i", file,
                "-vf", f"fps=1/{interval}",
                "-q:v", "2",
                output_pattern
            ]

            self.current_process = subprocess.Popen(cmd)

            while self.current_process.poll() is None:
                if self.stop_flag:
                    self.current_process.kill()
                    break

                produced = self.count_images()

                if self.total_expected > 0:
                    progress = produced / self.total_expected * 100
                    self.progress["value"] = progress

                    elapsed = time.time() - start_time
                    if produced > 0:
                        remaining = elapsed * (self.total_expected - produced) / produced
                        eta = datetime.now() + timedelta(seconds=remaining)
                        self.time_label.config(
                            text=f"終了予定時間: {eta.strftime('%Y-%m-%d %H:%M:%S')}"
                        )

                self.root.update()
                time.sleep(0.3)

        self.progress["value"] = 100
        self.time_label.config(text="完了")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
