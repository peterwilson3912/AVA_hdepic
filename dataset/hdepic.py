# Run:
# python graph_construction.py --model YOUR_MODEL --dataset hdepic --video_id 1 --gpus 1
#
# IDs 1..156      → single videos
# IDs 157..395    → multi-video combinations (treated as one concatenated video)
#
# To see which ID maps to which video:
#   python datas/hdepic/generate_index.py        # full regen
#   python datas/hdepic/generate_index.py --append  # add new videos only

import os
import json
from torch.utils.data import Dataset
from video_utils import VideoRepresentation, CombinedVideoRepresentation
from typing import Union

VIDEOS_ROOT = "/data2/yaxuanli/HD-EPIC/Videos"


class HDEpicDataset(Dataset):
    def __init__(
        self,
        index_file="datas/hdepic/hdepic_index.json",
        qa_file="datas/hdepic/hdepic_qa.json",
        work_path="AVA_cache/hdepic",
    ):
        self.work_path = work_path
        self.videos_root = VIDEOS_ROOT

        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"Index file not found: {index_file}\n"
                "Run: python datas/hdepic/generate_index.py"
            )
        if not os.path.exists(qa_file):
            raise FileNotFoundError(
                f"QA file not found: {qa_file}\n"
                "Run: python datas/hdepic/generate_index.py"
            )

        with open(index_file) as f:
            self.index = json.load(f)

        with open(qa_file) as f:
            self.qa_data = json.load(f)

        # id -> entry lookup
        self.id_map = {entry["id"]: entry for entry in self.index}

    def __len__(self):
        return len(self.index)

    def _video_path(self, video_name: str) -> str:
        participant = video_name.split("-")[0]
        return os.path.join(self.videos_root, participant, f"{video_name}.mp4")

    def get_video_info(self, video_id: Union[int, str]):
        if isinstance(video_id, str):
            video_id = int(video_id)

        if video_id not in self.id_map:
            raise KeyError(f"video_id {video_id} not found. Valid range: 1 to {len(self.index)}")

        entry = self.id_map[video_id]
        qas = self.qa_data.get(str(video_id), [])

        # Format choices into question text (consistent with other datasets)
        formatted_qas = []
        for qa in qas:
            choices = qa.get("choices", [])
            labels = [chr(ord("A") + i) for i in range(len(choices))]
            labeled = [f"{lbl}. {ch}" for lbl, ch in zip(labels, choices)]
            question_text = qa["question"]
            if labeled:
                question_text = question_text + "\n" + "\n".join(labeled)

            formatted_qas.append({
                "question_id": qa.get("question_id"),
                "task_type": qa.get("task_type", ""),
                "question": question_text,
                "answer": chr(ord("A") + qa["correct_idx"]),
                "correct_idx": qa["correct_idx"],
                "time_references": qa.get("time_references", []),
            })

        if entry["type"] == "single":
            video_name = entry["video_name"]
            return {
                "video_id": video_id,
                "type": "single",
                "video_name": video_name,
                "video_path": self._video_path(video_name),
                "qa": formatted_qas,
            }
        else:
            video_names = entry["video_names"]
            return {
                "video_id": video_id,
                "type": "combined",
                "video_names": video_names,
                "video_paths": [self._video_path(n) for n in video_names],
                "qa": formatted_qas,
            }

    def get_video(self, video_id: Union[int, str]):
        if isinstance(video_id, str):
            video_id = int(video_id)

        if video_id not in self.id_map:
            raise KeyError(f"video_id {video_id} not found. Valid range: 1 to {len(self.index)}")

        entry = self.id_map[video_id]

        if entry["type"] == "single":
            video_name = entry["video_name"]
            source_path = self._video_path(video_name)
            work_dir = os.path.join(self.work_path, video_name)
            os.makedirs(work_dir, exist_ok=True)
            return VideoRepresentation(source_path, work_dir)
        else:
            video_names = entry["video_names"]
            # Use a stable cache dir name: hash of sorted video names
            combo_key = "+".join(video_names)
            work_dir = os.path.join(self.work_path, "combined", combo_key)
            os.makedirs(work_dir, exist_ok=True)
            source_paths = [self._video_path(n) for n in video_names]
            return CombinedVideoRepresentation(source_paths, video_names, work_dir)
