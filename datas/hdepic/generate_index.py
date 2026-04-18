"""
Generate hdepic_index.json and hdepic_qa.json for the HD-EPIC dataset.

Usage:
    python generate_index.py                  # full regeneration
    python generate_index.py --append         # add new videos/combos to the end only

hdepic_index.json maps stable integer IDs to video files:
  - IDs 1..N: single videos (sorted by participant, then filename)
  - IDs N+1..: unique multi-video combinations (sorted by their video name tuples)

hdepic_qa.json maps integer ID (as string key) to a list of QA entries.
"""

import argparse
import glob
import json
import os

VIDEOS_ROOT = "/data2/yaxuanli/HD-EPIC/Videos"
ANN_DIR = "/data2/yaxuanli/HD-EPIC/hd-epic-annotations/vqa-benchmark"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(OUT_DIR, "hdepic_index.json")
QA_FILE = os.path.join(OUT_DIR, "hdepic_qa.json")
QUESTION_LOOKUP_FILE = os.path.join(OUT_DIR, "hdepic_question_lookup.json")


def scan_videos(videos_root):
    """Return sorted list of (participant, video_name) for all .mp4 files."""
    entries = []
    for participant in sorted(os.listdir(videos_root)):
        part_dir = os.path.join(videos_root, participant)
        if not os.path.isdir(part_dir) or not participant.startswith("P"):
            continue
        for fname in sorted(os.listdir(part_dir)):
            if fname.lower().endswith(".mp4"):
                video_name = os.path.splitext(fname)[0]
                entries.append((participant, video_name))
    return entries


def load_annotations(ann_dir):
    """
    Load all annotation files.
    Returns:
        single_qa: dict[video_name -> list of qa dicts]
        combo_qa:  dict[frozenset(video_names) -> list of qa dicts]
                   (keyed by frozenset so order doesn't matter for lookup,
                    but we also store the ordered tuple for stable sorting)
    Also returns:
        combo_ordered: dict[frozenset -> tuple of video_names in annotation order]
    """
    single_qa = {}   # video_name -> [qa, ...]
    combo_qa = {}    # frozenset(video_names) -> [qa, ...]
    combo_ordered = {}  # frozenset -> tuple of video_names (stable order from first seen)

    ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    for fpath in ann_files:
        task_type = os.path.splitext(os.path.basename(fpath))[0]
        with open(fpath) as f:
            data = json.load(f)

        for question_id, q in data.items():
            inputs = q["inputs"]
            # inputs is ordered: "video 1", "video 2", ...
            video_keys = sorted(inputs.keys(), key=lambda k: int(k.split()[-1]))
            video_names = [inputs[vk]["id"] for vk in video_keys]

            # Build time_references per video slot
            time_refs = []
            for vk in video_keys:
                vinfo = inputs[vk]
                ref = {"video_name": vinfo["id"]}
                if "start_time" in vinfo:
                    ref["start_time"] = vinfo["start_time"]
                if "end_time" in vinfo:
                    ref["end_time"] = vinfo["end_time"]
                time_refs.append(ref)

            qa_entry = {
                "question_id": question_id,
                "task_type": task_type,
                "question": q["question"],
                "choices": q["choices"],
                "correct_idx": q["correct_idx"],
                "time_references": time_refs,
            }
            if "others" in q:
                qa_entry["others"] = q["others"]

            if len(video_names) == 1:
                vname = video_names[0]
                single_qa.setdefault(vname, []).append(qa_entry)
            else:
                key = frozenset(video_names)
                combo_qa.setdefault(key, []).append(qa_entry)
                if key not in combo_ordered:
                    combo_ordered[key] = tuple(video_names)

    return single_qa, combo_qa, combo_ordered


def build_index_full(single_videos, combo_ordered):
    """Build index from scratch. Returns list of index entries."""
    index = []
    next_id = 1

    # Single videos first
    for participant, video_name in single_videos:
        index.append({
            "id": next_id,
            "type": "single",
            "video_name": video_name,
            "participant": participant,
        })
        next_id += 1

    # Multi-video combinations, sorted by their canonical tuple for stability
    for key in sorted(combo_ordered.keys(), key=lambda k: sorted(k)):
        ordered = combo_ordered[key]
        index.append({
            "id": next_id,
            "type": "combined",
            "video_names": list(ordered),
        })
        next_id += 1

    return index


def build_index_append(existing_index, single_videos, combo_ordered):
    """
    Append new entries to existing index without changing existing IDs.
    Returns updated index list.
    """
    existing_single = {e["video_name"] for e in existing_index if e["type"] == "single"}
    existing_combos = {
        frozenset(e["video_names"])
        for e in existing_index if e["type"] == "combined"
    }
    next_id = max(e["id"] for e in existing_index) + 1
    index = list(existing_index)

    for participant, video_name in single_videos:
        if video_name not in existing_single:
            index.append({
                "id": next_id,
                "type": "single",
                "video_name": video_name,
                "participant": participant,
            })
            next_id += 1

    for key in sorted(combo_ordered.keys(), key=lambda k: sorted(k)):
        if key not in existing_combos:
            ordered = combo_ordered[key]
            index.append({
                "id": next_id,
                "type": "combined",
                "video_names": list(ordered),
            })
            next_id += 1

    return index


def build_qa(index, single_qa, combo_qa):
    """Build qa dict keyed by string ID."""
    qa_out = {}
    for entry in index:
        eid = str(entry["id"])
        if entry["type"] == "single":
            qa_out[eid] = single_qa.get(entry["video_name"], [])
        else:
            key = frozenset(entry["video_names"])
            qa_out[eid] = combo_qa.get(key, [])
    return qa_out


def build_question_lookup(qa_out):
    """Build a flat lookup: question_id string -> {video_id, question_index}."""
    lookup = {}
    for vid_id, qlist in qa_out.items():
        for i, q in enumerate(qlist):
            qid = q.get("question_id")
            if qid:
                lookup[qid] = {"video_id": int(vid_id), "question_index": i}
    return lookup


def main():
    parser = argparse.ArgumentParser(description="Generate hdepic_index.json and hdepic_qa.json")
    parser.add_argument("--append", action="store_true",
                        help="Append new videos/combos to existing index without changing existing IDs")
    args = parser.parse_args()

    print("Scanning videos...")
    single_videos = scan_videos(VIDEOS_ROOT)
    print(f"  Found {len(single_videos)} single video files")

    print("Loading annotations...")
    single_qa, combo_qa, combo_ordered = load_annotations(ANN_DIR)
    print(f"  {sum(len(v) for v in single_qa.values())} single-video questions across {len(single_qa)} videos")
    print(f"  {sum(len(v) for v in combo_qa.values())} multi-video questions across {len(combo_ordered)} unique combinations")

    if args.append and os.path.exists(INDEX_FILE):
        print(f"Append mode: loading existing {INDEX_FILE}")
        with open(INDEX_FILE) as f:
            existing_index = json.load(f)
        index = build_index_append(existing_index, single_videos, combo_ordered)
        new_count = len(index) - len(existing_index)
        print(f"  Added {new_count} new entries (total: {len(index)})")
    else:
        if args.append:
            print(f"  (--append specified but {INDEX_FILE} not found; doing full generation)")
        index = build_index_full(single_videos, combo_ordered)
        print(f"  Built index with {len(index)} entries ({len(single_videos)} single + {len(combo_ordered)} combined)")

    print("Building QA mapping...")
    qa_out = build_qa(index, single_qa, combo_qa)
    total_q = sum(len(v) for v in qa_out.values())
    print(f"  {total_q} total questions mapped")

    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {INDEX_FILE}")

    with open(QA_FILE, "w") as f:
        json.dump(qa_out, f, indent=2)
    print(f"Wrote {QA_FILE}")

    print("Building question lookup...")
    question_lookup = build_question_lookup(qa_out)
    print(f"  {len(question_lookup)} unique question IDs indexed")
    with open(QUESTION_LOOKUP_FILE, "w") as f:
        json.dump(question_lookup, f, indent=2)
    print(f"Wrote {QUESTION_LOOKUP_FILE}")


if __name__ == "__main__":
    main()
