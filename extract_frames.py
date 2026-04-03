# Run:
# python extract_frames_ffmpeg.py \
#   /data/yaxuanli/AVA/datas/hdepic/videos/P01-20240202-110250.mp4 \
#   /data/yaxuanli/AVA/AVA_cache/hdepic/P01-20240202-110250/frames
#
# Example with custom FPS:
# python extract_frames_ffmpeg.py \
#   /data/yaxuanli/AVA/datas/hdepic/videos/P01-20240202-110250.mp4 \
#   /data/yaxuanli/AVA/AVA_cache/hdepic/P01-20240202-110250/frames \
#   --fps 2
#
# Example with custom FPS and output size:
# python extract_frames_ffmpeg.py \
#   /data/yaxuanli/AVA/datas/hdepic/videos/P01-20240202-110250.mp4 \
#   /data/yaxuanli/AVA/AVA_cache/hdepic/P01-20240202-110250/frames \
#   --fps 1 --width 512 --height 512
#
# Example keeping aspect ratio, set only width:
# python extract_frames_ffmpeg.py \
#   /data/yaxuanli/AVA/datas/hdepic/videos/P01-20240202-110250.mp4 \
#   /data/yaxuanli/AVA/AVA_cache/hdepic/P01-20240202-110250/frames \
#   --fps 1 --width 768





import argparse
import shutil
import subprocess
from pathlib import Path


def build_vf_string(fps: float | None, width: int | None, height: int | None) -> str | None:
    filters = []

    if fps is not None:
        filters.append(f"fps={fps}")

    if width is not None or height is not None:
        # -1 means ffmpeg auto-computes that dimension to preserve aspect ratio
        w = width if width is not None else -1
        h = height if height is not None else -1
        filters.append(f"scale={w}:{h}")

    if not filters:
        return None

    return ",".join(filters)


def extract_frames_with_ffmpeg(
    input_video_path: str,
    output_frames_dir: str,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
    overwrite: bool = False,
) -> None:
    input_video = Path(input_video_path)
    output_dir = Path(output_frames_dir)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not found in PATH.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "%d.jpg"

    vf_string = build_vf_string(fps=fps, width=width, height=height)

    cmd = ["ffmpeg"]

    cmd.append("-y" if overwrite else "-n")
    cmd.extend(["-i", str(input_video)])

    if vf_string is not None:
        cmd.extend(["-vf", vf_string])

    cmd.extend([
        "-start_number", "0",
        str(output_pattern),
    ])

    print("Running command:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed.\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    print(f"Done. Frames saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract video frames with ffmpeg.")
    parser.add_argument("input_video_path", type=str, help="Path to input video file")
    parser.add_argument("output_frames_dir", type=str, help="Path to output frames folder")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output frame rate. Example: --fps 1 means 1 frame per second. "
             "If omitted, extract all original frames.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width. If only width is set, height is auto-computed.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height. If only height is set, width is auto-computed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output frames if they already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frames_with_ffmpeg(
        input_video_path=args.input_video_path,
        output_frames_dir=args.output_frames_dir,
        fps=args.fps,
        width=args.width,
        height=args.height,
        overwrite=args.overwrite,
    )