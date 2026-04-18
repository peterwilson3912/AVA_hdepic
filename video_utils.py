import json
import os
import cv2
from PIL import Image
from tqdm import tqdm

class VideoRepresentation:
    def __init__(self, video_source_path, work_dir):
        self.video_source_path = video_source_path
        self.work_dir = work_dir
        self.frames_dir = os.path.join(work_dir, "frames")
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        
        if not os.path.exists(os.path.join(work_dir, "config.json")):
            self.extract_frames()
        else:
            self.config = json.load(open(os.path.join(work_dir, "config.json")))
    
    def extract_frames(self):
        config = read_video_frames(self.video_source_path, self.frames_dir)
        if not config:
            print(f"Failed to extract frames from {self.video_source_path}.")
            return
        self.config = config
        
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        
        with open(os.path.join(self.work_dir, "config.json"), "w") as f:
            json.dump(config, f)
    
    def get_frames_by_fps_multiple(self, fps=0, durations: list = None):
        if durations is None:
            durations = [(0, self.config["duration"])]

        durations = sorted(durations, key=lambda x: x[0])

        all_frames = []
        all_timestamps = []
        all_frame_indices = []

        for duration in durations:
            frames, timestamps, frame_indices = self.get_frames_by_fps(fps=fps, duration=duration)
            all_frames.extend(frames)
            all_timestamps.extend(timestamps)
            all_frame_indices.extend(frame_indices)

        return all_frames, all_timestamps, all_frame_indices
    
    def get_frames_by_fps(self, fps=0, duration: tuple = None):
        if duration is None:
            duration = (0, self.config["duration"])

        start_time, end_time = duration
        video_fps = self.config["fps"]
        frame_interval = int(video_fps / fps) if fps > 0 else 1

        if start_time == end_time:
            start_frame = int(start_time * video_fps)
            selected_frames = [start_frame]
        else:
            start_frame = int(start_time * video_fps)
            end_frame = int(end_time * video_fps)
            selected_frames = range(start_frame, end_frame, frame_interval)

        frames = []
        timestamps = []
        frame_indices = []
        for frame_idx in selected_frames:
            frame_time = frame_idx / video_fps
            frame_path = os.path.join(self.frames_dir, f"{frame_idx}.jpg")
            if os.path.exists(frame_path):
                frames.append(Image.open(frame_path))
                timestamps.append(frame_time)
                frame_indices.append(frame_idx)

        return frames, timestamps, frame_indices
    
    def get_frames_by_num(self, num_frames=1, duration: tuple = None):
        if duration is None:
            duration = (0, self.config["duration"])

        start_time, end_time = duration
        video_fps = self.config["fps"]

        if start_time == end_time:
            end_time += 1

        if num_frames == 1:
            time_points = [(start_time + end_time) / 2]
        elif num_frames == 2:
            time_points = [start_time, end_time]
        else:
            time_points = [
                start_time + i * (end_time - start_time) / (num_frames - 1)
                for i in range(num_frames)
            ]

        frames = []
        timestamps = []
        frame_indices = []

        for frame_time in time_points:
            frame_idx = int(frame_time * video_fps)
            frame_path = os.path.join(self.frames_dir, f"{frame_idx}.jpg")
            if os.path.exists(frame_path):
                frames.append(Image.open(frame_path))
                timestamps.append(frame_time)
                frame_indices.append(frame_idx)

        return frames, timestamps, frame_indices
    
    def get_frames_by_timestamps(self, timestamps):
        video_fps = self.config["fps"]
        frames = []
        for timestamp in timestamps:
            frame_idx = int(timestamp * video_fps)
            frame_path = os.path.join(self.frames_dir, f"{frame_idx}.jpg")
            if os.path.exists(frame_path):
                frames.append(Image.open(frame_path))
        
        return frames
    
    def get_frames_by_indices(self, frame_indices):
        frames = []
        for frame_idx in frame_indices:
            frame_path = os.path.join(self.frames_dir, f"{frame_idx}.jpg")
            if os.path.exists(frame_path):
                frames.append(Image.open(frame_path))
        
        return frames

class CombinedVideoRepresentation:
    """
    Treats multiple video files as a single concatenated video.
    Frames are extracted from each component video with offset indices so the
    combined frame sequence is continuous. Timestamps for each component are
    offset by the cumulative duration of all preceding components.

    config.json stores:
      fps, duration, frame_count (same keys as VideoRepresentation),
      segments: list of {video_name, source_path, fps, duration, frame_count, frame_offset, time_offset}
    """

    def __init__(self, video_source_paths: list, video_names: list, work_dir: str):
        self.video_source_paths = video_source_paths
        self.video_names = video_names
        self.work_dir = work_dir
        self.frames_dir = os.path.join(work_dir, "frames")
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)

        config_path = os.path.join(work_dir, "config.json")
        if not os.path.exists(config_path):
            self.extract_frames()
        else:
            self.config = json.load(open(config_path))

    def extract_frames(self):
        segments = []
        frame_offset = 0
        time_offset = 0.0

        for video_path, video_name in zip(self.video_source_paths, self.video_names):
            print(f"Extracting frames from {video_name}...")
            seg_config = read_video_frames(video_path, self.frames_dir, frame_offset=frame_offset)
            if not seg_config:
                print(f"Failed to extract frames from {video_path}.")
                return
            segments.append({
                "video_name": video_name,
                "source_path": video_path,
                "fps": seg_config["fps"],
                "duration": seg_config["duration"],
                "frame_count": seg_config["frame_count"],
                "frame_offset": frame_offset,
                "time_offset": time_offset,
            })
            frame_offset += seg_config["frame_count"]
            time_offset += seg_config["duration"]

        # Use fps of first segment as the canonical fps.
        # Keys "fps", "duration", "frame_count" match VideoRepresentation so
        # all downstream AVA code (events.py, operate.py, etc.) works unchanged.
        canonical_fps = segments[0]["fps"] if segments else 1.0
        self.config = {
            "fps": canonical_fps,
            "duration": time_offset,
            "frame_count": frame_offset,
            "segments": segments,
        }
        with open(os.path.join(self.work_dir, "config.json"), "w") as f:
            json.dump(self.config, f)

    def _segment_for_video_name(self, video_name: str):
        """Return segment dict for a given video_name."""
        for seg in self.config["segments"]:
            if seg["video_name"] == video_name:
                return seg
        raise KeyError(f"Video name {video_name} not found in combined video segments")

    def time_offset_for_video(self, video_name: str) -> float:
        """Return the absolute time offset (seconds) of a component video within the combined video."""
        return self._segment_for_video_name(video_name)["time_offset"]

    def get_frames_by_fps(self, fps=0, duration: tuple = None):
        if duration is None:
            duration = (0, self.config["duration"])

        start_time, end_time = duration
        video_fps = self.config["fps"]
        frame_interval = int(video_fps / fps) if fps > 0 else 1

        if start_time == end_time:
            start_frame = int(start_time * video_fps)
            selected_frames = [start_frame]
        else:
            start_frame = int(start_time * video_fps)
            end_frame = int(end_time * video_fps)
            selected_frames = range(start_frame, end_frame, frame_interval)

        frames = []
        timestamps = []
        frame_indices = []
        for frame_idx in selected_frames:
            frame_time = frame_idx / video_fps
            frame_path = os.path.join(self.frames_dir, f"{frame_idx}.jpg")
            if os.path.exists(frame_path):
                frames.append(Image.open(frame_path))
                timestamps.append(frame_time)
                frame_indices.append(frame_idx)

        return frames, timestamps, frame_indices

    def get_frames_by_fps_multiple(self, fps=0, durations: list = None):
        if durations is None:
            durations = [(0, self.config["duration"])]

        durations = sorted(durations, key=lambda x: x[0])
        all_frames, all_timestamps, all_frame_indices = [], [], []
        for duration in durations:
            frames, timestamps, frame_indices = self.get_frames_by_fps(fps=fps, duration=duration)
            all_frames.extend(frames)
            all_timestamps.extend(timestamps)
            all_frame_indices.extend(frame_indices)
        return all_frames, all_timestamps, all_frame_indices

    def get_frames_by_num(self, num_frames=1, duration: tuple = None):
        if duration is None:
            duration = (0, self.config["duration"])

        start_time, end_time = duration
        video_fps = self.config["fps"]

        if start_time == end_time:
            end_time += 1

        if num_frames == 1:
            time_points = [(start_time + end_time) / 2]
        elif num_frames == 2:
            time_points = [start_time, end_time]
        else:
            time_points = [
                start_time + i * (end_time - start_time) / (num_frames - 1)
                for i in range(num_frames)
            ]

        frames, timestamps, frame_indices = [], [], []
        for frame_time in time_points:
            frame_idx = int(frame_time * video_fps)
            frame_path = os.path.join(self.frames_dir, f"{frame_idx}.jpg")
            if os.path.exists(frame_path):
                frames.append(Image.open(frame_path))
                timestamps.append(frame_time)
                frame_indices.append(frame_idx)

        return frames, timestamps, frame_indices

    def get_frames_by_timestamps(self, timestamps):
        video_fps = self.config["fps"]
        frames = []
        for timestamp in timestamps:
            frame_idx = int(timestamp * video_fps)
            frame_path = os.path.join(self.frames_dir, f"{frame_idx}.jpg")
            if os.path.exists(frame_path):
                frames.append(Image.open(frame_path))
        return frames

    def get_frames_by_indices(self, frame_indices):
        frames = []
        for frame_idx in frame_indices:
            frame_path = os.path.join(self.frames_dir, f"{frame_idx}.jpg")
            if os.path.exists(frame_path):
                frames.append(Image.open(frame_path))
        return frames


def read_video_frames(video_path, save_path, target_fps=0, target_resolution=(540, 360), chunk_size=1000, frame_offset=0):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return [], {}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = video_frame_count / video_fps

    frame_interval = int(video_fps / target_fps) if target_fps > 0 else 1

    frame_count = 0

    with tqdm(total=video_frame_count, desc="Extracting video frames") as pbar:
        while True:
            for _ in range(chunk_size):
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    if target_resolution is not None:
                        frame = cv2.resize(frame, target_resolution)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    image.save(os.path.join(save_path, f"{frame_offset + frame_count}.jpg"))

                frame_count += 1
                pbar.update(1)

            if not ret:
                break

    cap.release()

    config = {
        "source_path": video_path,
        "fps": video_fps,
        "width": video_width,
        "height": video_height,
        "frame_count": video_frame_count,
        "duration": video_duration
    }

    return config