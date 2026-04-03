# Run:
# python graph_construction.py --model YOUR_MODEL --dataset hdepic --video_id 1 --gpus 1

import os
import json
from torch.utils.data import Dataset
from video_utils import VideoRepresentation
from typing import Union


class HDEpicDataset(Dataset):
    def __init__(
        self,
        json_file="datas/hdepic/hdepic.json",
        videos_path="datas/hdepic/videos",
        work_path="AVA_cache/hdepic",
    ):
        self.videos_path = videos_path
        self.work_path = work_path
        self.json_file = json_file

        if not os.path.exists(self.videos_path):
            raise FileNotFoundError(f"videos_path does not exist: {self.videos_path}")

        self.video_files = sorted([
            f for f in os.listdir(self.videos_path)
            if f.lower().endswith(".mp4")
        ])

        if len(self.video_files) == 0:
            raise RuntimeError(f"No .mp4 files found in: {self.videos_path}")

        if not os.path.exists(self.json_file):
            raise FileNotFoundError(f"json_file does not exist: {self.json_file}")

        with open(self.json_file, "r") as f:
            self.qa_data = json.load(f)

        # Build a map: video_id -> qa entry
        self.qa_dict = {}
        for item in self.qa_data:
            vid = int(item["video_id"])
            self.qa_dict[vid] = item

    def __len__(self):
        return len(self.video_files)

    def get_video_info(self, video_id: Union[int, str]):
        if isinstance(video_id, str):
            video_id = int(video_id)

        idx = video_id - 1
        if idx < 0 or idx >= len(self.video_files):
            raise IndexError(
                f"video_id {video_id} out of range. Valid range: 1 to {len(self.video_files)}"
            )

        video_filename = self.video_files[idx]
        video_name = os.path.splitext(video_filename)[0]

        if video_id not in self.qa_dict:
            raise KeyError(f"No QA found for video_id {video_id} in {self.json_file}")

        qas = self.qa_dict[video_id]["qa"]

        # Match VideoMME behavior: append options into question text
        formatted_qas = []
        for qa in qas:
            formatted_qa = qa.copy()

            question = formatted_qa["question"]
            options = formatted_qa.get("options", [])

            if len(options) > 0:
                formatted_qa["question"] = question + "\n" + "\n".join(options)

            formatted_qas.append({
                "question": formatted_qa["question"],
                "answer": formatted_qa["answer"],
                "task_type": formatted_qa.get("task_type", ""),
                "question_id": formatted_qa.get("question_id", None),
            })

        return {
            "video_id": video_id,
            "video_name": video_name,
            "video_path": os.path.join(self.videos_path, video_filename),
            "qa": formatted_qas,
        }

    def get_video(self, video_id: Union[int, str]):
        if isinstance(video_id, str):
            video_id = int(video_id)

        idx = video_id - 1
        if idx < 0 or idx >= len(self.video_files):
            raise IndexError(
                f"video_id {video_id} out of range. Valid range: 1 to {len(self.video_files)}"
            )

        video_filename = self.video_files[idx]
        video_name = os.path.splitext(video_filename)[0]

        source_path = os.path.join(self.videos_path, video_filename)
        work_path = os.path.join(self.work_path, video_name)

        if not os.path.exists(work_path):
            os.makedirs(work_path)

        return VideoRepresentation(source_path, work_path)