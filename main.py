"""Command-line entry point for video processing and analytics workflows."""

import argparse
from pathlib import Path

from config import PROCESSED_DIR, ensure_project_directories
from detection.detector import YOLODetector
from detection.video_processor import process_video
from analytics.dataset_generation import generate_dataset
from models.pipeline import load_dataset, train_and_compare
from utils.logging_config import get_logger

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Football video detection and predictive analytics pipeline."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    process = commands.add_parser("process-video", help="Run YOLO on a video and export detections.")
    process.add_argument("--video", required=True, help="Path to a local match video.")
    process.add_argument("--model", default=None, help="Path to an existing Ultralytics YOLO model.")
    process.add_argument("--output", default=None, help="Detection CSV output path.")
    process.add_argument("--sample-every", type=int, default=1, help="Process every Nth frame.")
    process.add_argument("--max-frames", type=int, default=None, help="Optional sampled-frame limit.")

    dataset = commands.add_parser("generate-dataset", help="Aggregate a detection CSV by frame.")
    dataset.add_argument("--detections", required=True, help="Detection CSV from process-video.")
    dataset.add_argument("--output", default=None, help="Generated dataset CSV path.")
    dataset.add_argument("--sqlite-table", default=None, help="Optional SQLite table name.")

    train = commands.add_parser("train", help="Train and compare regressors from a dataset CSV.")
    train.add_argument("--dataset", required=True, help="Frame-level dataset CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_project_directories()
    try:
        if args.command == "process-video":
            model_path = args.model
            detector = YOLODetector(**({"model_path": model_path} if model_path else {}))
            output = args.output or PROCESSED_DIR / f"{Path(args.video).stem}_detections.csv"
            result = process_video(
                args.video,
                detector,
                output_csv=output,
                sample_every=args.sample_every,
                max_frames=args.max_frames,
            )
            print(f"Saved {len(result)} detection rows to {output}")
        elif args.command == "generate-dataset":
            result = generate_dataset(args.detections, args.output, args.sqlite_table)
            print(f"Generated {len(result)} frame rows.")
        elif args.command == "train":
            metrics, _ = train_and_compare(load_dataset(args.dataset))
            print(metrics.to_string(index=False))
    except (FileNotFoundError, ImportError, ValueError, OSError, RuntimeError):
        LOGGER.exception("Pipeline command failed.")
        raise


if __name__ == "__main__":
    main()
