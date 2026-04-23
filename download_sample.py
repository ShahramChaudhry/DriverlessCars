"""
Download a short public-domain driving video for testing.
Usage:  python download_sample.py
"""
import os
import urllib.request

# Public test video from Google's open media bucket.
# Replace with any front-camera .mp4 clip (BDD100K, nuScenes, dashcam footage, etc.)
SAMPLE_URL = (
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/videos/zidane.mp4"
)
OUT_DIR  = "sample_videos"
OUT_FILE = os.path.join(OUT_DIR, "driving_sample.mp4")


def download_sample(url: str = SAMPLE_URL, out_file: str = OUT_FILE) -> str:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if os.path.exists(out_file):
        print(f"Already downloaded: {out_file}")
        return out_file
    print(f"Downloading sample video -> {out_file} ...")
    urllib.request.urlretrieve(url, out_file, reporthook=_progress)
    print("\nDone.")
    return out_file


def _progress(block, block_size, total):
    downloaded = block * block_size
    if total > 0:
        pct = min(100, downloaded * 100 // total)
        print(f"\r  {pct}%  ({downloaded // 1024} KB)", end="", flush=True)


if __name__ == "__main__":
    path = download_sample()
    print(f"\nSample saved to: {path}")
    print("Upload it in the Gradio demo, or test loading with:")
    print(f"  python -c \"from inference import load_video_frames; "
          f"r = load_video_frames('{path}'); print(len(r[0]), 'frames loaded')\"")
