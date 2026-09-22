#!/usr/bin/env python
"""
Identity scoring with a face-recognition embedding.

Why this exists: DINOv2 on a face crop responds to pose, expression, lighting
and hair as well as identity, so it answers "do these faces look alike?" rather
than "is this the same person?". SFace is trained for face recognition
specifically, with a loss that pulls the same identity together and pushes
different identities apart, so its cosine similarity is a direct identity
measure. This is the metric InstantID and PhotoMaker report, in the same family
as ArcFace.

Pipeline per image: YuNet detects faces, the largest is aligned to a canonical
5-point layout, SFace embeds it into 128 dimensions. Alignment matters: it
removes in-plane rotation and scale, which is part of why this is less
pose-sensitive than DINO on a raw crop.

OpenCV's reference cosine threshold for "same person" with SFace is 0.363, so
the fraction of pairs above it is reported alongside the mean.

Usage:
    python eval/face_identity.py --out_dir results/baseline
    python eval/face_identity.py --dir data/training_set_s065
    python eval/face_identity.py --compare results/baseline/eval_sface.json \
                                           results/lora_desc/eval_sface.json
"""

import argparse
import itertools
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

import cv2
import numpy as np

ARMS = ["prompt_only", "fixed_seed", "high_cfg"]
SAME_ID_THRESHOLD = 0.363   # OpenCV's reference cosine threshold for SFace

MODELS = {
    "detector": (
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx",
    ),
    "recognizer": (
        "face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx",
    ),
}


def fetch_models(model_dir="models"):
    """Download the ONNX models once, rejecting Git LFS pointer files.

    raw.githubusercontent.com serves a ~130 byte text pointer for LFS-tracked
    files instead of the model, which fails later with a confusing ONNX parse
    error. The github.com/.../raw/... URL follows the redirect to the real blob.
    """
    os.makedirs(model_dir, exist_ok=True)
    paths = {}
    for key, (name, url) in MODELS.items():
        path = os.path.join(model_dir, name)
        if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, path)
        size = os.path.getsize(path)
        if size < 1_000_000:
            head = open(path, "rb").read(40)
            os.remove(path)
            sys.exit(
                f"{name} downloaded as {size} bytes starting {head!r}. That is a Git "
                "LFS pointer, not the model. Download it in a browser from the "
                f"opencv_zoo repository and place it at {path}."
            )
        paths[key] = path
    return paths


class FaceScorer:
    def __init__(self, model_dir="models"):
        paths = fetch_models(model_dir)
        self.detector = cv2.FaceDetectorYN.create(
            paths["detector"], "", (320, 320), score_threshold=0.7
        )
        self.recognizer = cv2.FaceRecognizerSF.create(paths["recognizer"], "")

    def embed(self, path):
        """Return a normalized 128-d identity embedding, or None if no face."""
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        if faces is None or len(faces) == 0:
            return None
        # faces rows: x, y, w, h, 5 landmark pairs, score. Take the largest box.
        largest = max(faces, key=lambda f: f[2] * f[3])
        aligned = self.recognizer.alignCrop(img, largest)
        feat = self.recognizer.feature(aligned).flatten()
        return feat / np.linalg.norm(feat)

    def score_folder(self, folder):
        exts = (".png", ".jpg", ".jpeg", ".webp")
        names = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
        embeddings, missing = [], []
        for name in names:
            e = self.embed(os.path.join(folder, name))
            (missing if e is None else embeddings).append(name if e is None else e)
        if len(embeddings) < 2:
            return None
        sims = [
            float(a @ b) for a, b in itertools.combinations(embeddings, 2)
        ]
        return {
            "n_faces": len(embeddings),
            "n_images": len(names),
            "images_without_face": missing,
            "sface_identity_mean": round(float(np.mean(sims)), 4),
            "sface_identity_worst_pair": round(float(np.min(sims)), 4),
            "pairs_above_same_id_threshold": round(
                float(np.mean([s >= SAME_ID_THRESHOLD for s in sims])), 3
            ),
            "per_pair": [round(s, 4) for s in sims],
        }


def print_table(results):
    header = f"{'arm':<14}{'SFace mean':>12}{'worst pair':>12}{'pairs >0.363':>14}{'faces':>8}"
    print("\n" + header)
    print("-" * len(header))
    for arm, r in results.items():
        print(
            f"{arm:<14}{r['sface_identity_mean']:>12.4f}"
            f"{r['sface_identity_worst_pair']:>12.4f}"
            f"{r['pairs_above_same_id_threshold']:>14.3f}"
            f"{r['n_faces']:>5}/{r['n_images']}"
        )
    print()


def compare(a_path, b_path):
    a = json.load(open(a_path))["arms"]
    b = json.load(open(b_path))["arms"]
    print(f"\n{a_path}  ->  {b_path}\n")
    for arm in a:
        if arm not in b:
            continue
        print(f"  {arm}")
        for k in ("sface_identity_mean", "sface_identity_worst_pair",
                  "pairs_above_same_id_threshold"):
            d = b[arm][k] - a[arm][k]
            print(f"    {k:<34}{a[arm][k]:>8.4f} -> {b[arm][k]:>8.4f}  ({d:+.4f})")
        print()
    print("  15 pairs per arm: treat small gaps as inconclusive.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", help="results directory containing arm subfolders")
    ap.add_argument("--dir", help="score a single flat folder of images")
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    scorer = FaceScorer(args.model_dir)

    if args.dir:
        print(json.dumps(scorer.score_folder(args.dir), indent=2))
        return

    if not args.out_dir:
        sys.exit("Pass --out_dir, --dir or --compare.")

    results = {}
    for arm in ARMS:
        arm_dir = os.path.join(args.out_dir, arm)
        if os.path.isdir(arm_dir):
            r = scorer.score_folder(arm_dir)
            if r:
                results[arm] = r

    out_json = os.path.join(args.out_dir, "eval_sface.json")
    with open(out_json, "w") as f:
        json.dump({
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "metric": "SFace cosine similarity on YuNet-aligned faces",
            "same_id_threshold": SAME_ID_THRESHOLD,
            "arms": results,
        }, f, indent=2)

    print_table(results)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
