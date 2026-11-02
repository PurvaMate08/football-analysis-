# Automatic Football Analysis and Predictive Sports Analytics System

A modular Python project for processing football video with an existing YOLO model, exporting observed detections to CSV/SQLite, generating frame-level datasets, analyzing those observations, training regression models, and exploring the results in Streamlit.

> **No example results are bundled.** All datasets, charts, reports, and model metrics are created from videos and detections processed by the user. Put videos and model weights in the project locally; large/private video and model files are ignored by Git.

## Project structure

```text
TY PROJECT/
├── .vscode/
│   ├── launch.json
│   └── settings.json
├── analytics/
│   ├── correlation.py
│   ├── dataset_generation.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── outlier_detection.py
│   ├── preprocessing.py
│   └── statistical_analysis.py
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── common.py
│       ├── dataset_explorer.py
│       ├── prediction.py
│       ├── reports.py
│       ├── statistical_analysis.py
│       └── video_processing.py
├── data/
│   ├── datasets/          # generated frame-level datasets
│   ├── generated_csv/     # optional user-generated/curated CSVs
│   ├── processed/         # per-object detection exports
│   ├── raw/               # optional local source videos
│   └── sample_videos/     # optional user-provided examples
├── detection/
│   ├── ball_detector.py
│   ├── detector.py
│   ├── frame_extractor.py
│   ├── player_detector.py
│   ├── results.py
│   └── video_processor.py
├── models/
│   ├── evaluation.py
│   ├── model_comparison.py
│   ├── persistence.py
│   ├── pipeline.py
│   ├── saved/             # locally trained Joblib bundles
│   └── train.py
├── reports/
│   ├── charts/            # exported Matplotlib/Seaborn charts
│   ├── generated/         # analytical JSON summaries
│   ├── model_performance/ # holdout metric CSV
│   └── report_generator.py
├── utils/
│   ├── config.py
│   ├── database.py
│   ├── file_handler.py
│   └── logging_config.py
├── visualization/
│   ├── analytics_plots.py
│   ├── detection_plots.py
│   └── static_plots.py
├── tests/
│   └── test_dataset_generation.py
├── app.py
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

### Folder and module guide

- **`detection/`** — `detector.py` wraps Ultralytics YOLO while allowing an already-loaded model to be injected; `frame_extractor.py` validates and samples OpenCV video frames; `player_detector.py` and `ball_detector.py` expose reusable class-specific calls; `results.py` defines the typed detection record; `video_processor.py` runs inference and writes a per-object CSV.
- **`data/`** — `raw/` is for optional local source videos; `sample_videos/` is for user-provided examples; `processed/` contains detection CSVs; `datasets/` contains frame-level analytical CSVs; `generated_csv/` is available for additional exports. Empty folders are tracked, but no videos or generated results are fabricated.
- **`analytics/`** — `dataset_generation.py` aggregates actual detections by observed frame; `preprocessing.py` validates schemas; `feature_engineering.py` creates next-observed-frame prediction features; `eda.py` computes descriptive summaries; `statistical_analysis.py` provides Welch testing, confidence intervals, and Statsmodels OLS trend estimates; `correlation.py` and `outlier_detection.py` provide correlation and IQR flags.
- **`models/`** — `train.py` fits Linear Regression, Decision Tree, and Random Forest regressors with a chronological holdout; `evaluation.py` computes MAE/RMSE/R²; `model_comparison.py` ranks metrics; `pipeline.py` coordinates feature creation, training, and exports; `persistence.py` saves/loads Joblib bundles in `saved/`.
- **`visualization/`** — `detection_plots.py` provides interactive Plotly player, ball, confidence, and spatial charts; `analytics_plots.py` provides correlation and actual-vs-predicted figures; `static_plots.py` exports real-data PNG charts with Matplotlib/Seaborn.
- **`dashboard/`** — `app.py` provides the basic single-page Streamlit interface: upload a video, analyze it, and view observed detection statistics, a player trend chart, and CSV downloads. Additional page modules remain available for deeper dataset exploration, statistics, predictions, and reports.
- **`reports/`** — `report_generator.py` writes measured summary JSON and static chart PNGs. `generated/`, `model_performance/`, and `charts/` store analytical summaries, model metrics, and charts.
- **`utils/`** — `config.py` re-exports root settings; `file_handler.py` validates paths and CSVs; `database.py` persists/loads DataFrames with SQLite; `logging_config.py` configures file and console logs.
- **Root files** — `config.py` centralizes paths and YOLO settings; `main.py` is the CLI; `app.py` is the Streamlit entry point; `.gitignore` excludes local data, logs, weights, and model artifacts; `.vscode/` contains launch/interpreter defaults; `requirements.txt` lists dependencies; `tests/` covers core dataset behavior.

## Data flow and interpretation

1. **Video → detections:** OpenCV extracts frames and `YOLODetector` runs the configured existing YOLO model. The output CSV has one row per object and includes video name, frame number/time, class, confidence, and bounding-box/center coordinates. A `class_id=-1` / `class_name=no_detection` row marks sampled frames with no detected objects.
2. **Detections → dataset:** `generate_dataset` aggregates observed frames into player count, ball-detected flag, mean confidence, and mean detected-object center. Explicit no-detection rows become observed zero-count frames. The pipeline does not infer detections for frames that were not sampled.
3. **Dataset → analysis/reports:** preprocessing and feature modules feed EDA, statistical analysis, correlations, outlier flags, and report/chart exports. Optional SQLite persistence is supported.
4. **Dataset → prediction:** models use current observed-frame features to predict the player count at the next **observed** frame in the same clip. The final portion of ordered observations is held out for evaluation; metrics are computed from that holdout, not generated as examples.
5. **Dashboard:** Streamlit accepts an uploaded video, runs the same detection/dataset pipeline, and displays actual sampled-frame statistics and a player-count chart.

The default class IDs in `config.py` assume COCO (`person=0`, `sports ball=32`). If your existing YOLO model uses other labels, update `PLAYER_CLASS_IDS` and `BALL_CLASS_IDS` before processing. The starter system counts detections; it does not yet track persistent player identities or infer match events/scores.

## Requirements and installation (Windows / PowerShell)

Use Python **3.10 or later**. From the project root in VS Code:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, open this folder and select `.venv` as the Python interpreter. If script activation is restricted by your local PowerShell policy, use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Provide a compatible local Ultralytics model file (for example, `yolov8n.pt`) or set `YOLO_MODEL_PATH`. The starter does not download weights automatically. For example:

```powershell
$env:YOLO_MODEL_PATH = "C:\path\to\your\existing-model.pt"
```

OpenCV, Pandas, NumPy, Matplotlib, Seaborn, Plotly, SciPy, Statsmodels, scikit-learn, Streamlit, and Joblib are installed from `requirements.txt`. SQLite is included with Python.

## Run the application

### Streamlit dashboard

```powershell
streamlit run app.py
```

Upload a supported video (`mp4`, `avi`, `mov`, `mkv`, `mpeg`, or `mpg`), choose a frame sampling interval, and click **Analyze video**. The app displays analyzed-frame count, total player detections, average and maximum detected players per frame, percentage of analyzed frames with a ball detection, mean YOLO confidence, and a player-count timeline. It also provides the generated CSV for download. Detection CSVs are written to `data/processed/`, frame datasets to `data/datasets/`, reports to `reports/`, and the temporary upload is deleted after processing.

### Command-line workflow

```powershell
# 1. Detect objects in an actual video
python main.py process-video --video "data\sample_videos\match.mp4" --model "C:\path\to\model.pt" --sample-every 5

# 2. Aggregate the detection CSV and optionally save it to SQLite
python main.py generate-dataset --detections "data\processed\match_detections.csv" --sqlite-table match_frames

# 3. Train and compare all three regression models
python main.py train --dataset "data\datasets\match_detections_frames.csv"
```

`--output` can be supplied to either of the first two commands to choose a CSV destination. `--max-frames` limits the number of sampled frames for a short validation run. At least five model-ready next-frame observations are required to train/evaluate.

## Validation

Run the focused tests after installing requirements:

```powershell
python -m pytest
```

The first video run can be compute-intensive. Use a short clip and a larger `--sample-every` value for initial validation. A valid detection CSV requires the expected columns and a readable video/model; failures are reported rather than replaced with synthetic results.
