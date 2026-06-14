"""Thai sign data-collection helper."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def save_clip(out_root: str, word: str, raw_seq: np.ndarray, source: str) -> tuple[str, str]:
    raw = np.asarray(raw_seq, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (543, 3):
        raise ValueError(f"raw_seq must be (T, 543, 3); got {raw.shape}")
    word_dir = os.path.join(out_root, word)
    os.makedirs(word_dir, exist_ok=True)
    existing = [f for f in os.listdir(word_dir) if f.endswith(".npy")]
    idx = len(existing)
    npy_path = os.path.join(word_dir, f"{idx}.npy")
    json_path = os.path.join(word_dir, f"{idx}.json")
    np.save(npy_path, raw)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"word": word, "source": source}, f, ensure_ascii=False)
    return npy_path, json_path


def main() -> None:  # pragma: no cover
    import importlib

    cv2 = importlib.import_module("cv2")
    mp = importlib.import_module("mediapipe")
    from tsl.features.landmarks import extract_sequence

    parser = argparse.ArgumentParser(description="Record Thai sign clips")
    parser.add_argument("--word", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-root", default=os.path.join("data", "thai"))
    parser.add_argument("--clips", type=int, default=6)
    parser.add_argument("--frames", type=int, default=48)
    args = parser.parse_args()
    cap = cv2.VideoCapture(0)
    with mp.solutions.holistic.Holistic() as holistic:
        for c in range(args.clips):
            input(f"[{c + 1}/{args.clips}] press Enter, then sign '{args.word}'...")
            frames = []
            while len(frames) < args.frames:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frames.append(frame_bgr)
            raw_seq = extract_sequence(holistic, frames)
            npy_path, _ = save_clip(args.out_root, args.word, raw_seq, args.source)
            print(f"saved {npy_path}")
    cap.release()


if __name__ == "__main__":  # pragma: no cover
    main()
