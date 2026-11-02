import pandas as pd

from analytics.dataset_generation import build_frame_dataset
from analytics.feature_engineering import create_prediction_features
from models.train import train_models


def test_frame_dataset_keeps_observed_no_detection_frames():
    detections = pd.DataFrame(
        [
            {
                "video_name": "clip.mp4", "frame_index": 0, "timestamp_seconds": 0.0,
                "class_id": 0, "class_name": "person", "confidence": 0.9,
                "center_x": 10.0, "center_y": 20.0,
            },
            {
                "video_name": "clip.mp4", "frame_index": 1, "timestamp_seconds": 0.1,
                "class_id": -1, "class_name": "no_detection", "confidence": float("nan"),
                "center_x": float("nan"), "center_y": float("nan"),
            },
        ]
    )

    dataset = build_frame_dataset(detections)

    assert dataset["player_count"].tolist() == [1, 0]
    assert dataset["ball_detected"].tolist() == [0, 0]


def test_features_predict_next_observed_player_count():
    dataset = pd.DataFrame(
        [
            {
                "video_name": "clip.mp4", "frame_index": index,
                "timestamp_seconds": index / 10, "player_count": count,
                "ball_detected": 0, "mean_confidence": 0.8,
                "mean_center_x": 10.0, "mean_center_y": 15.0,
            }
            for index, count in enumerate([2, 3, 4])
        ]
    )

    features = create_prediction_features(dataset)

    assert features["target_player_count"].tolist() == [3, 4]
    assert features["frame_delta"].tolist() == [0, 1]


def test_training_compares_all_configured_regressors():
    dataset = pd.DataFrame(
        [
            {
                "video_name": "clip.mp4", "frame_index": index,
                "timestamp_seconds": index / 10, "player_count": index % 5,
                "ball_detected": index % 2, "mean_confidence": 0.6 + index / 100,
                "mean_center_x": index * 10.0, "mean_center_y": index * 5.0,
            }
            for index in range(12)
        ]
    )

    metrics, fitted = train_models(create_prediction_features(dataset))

    assert set(fitted) == {"linear_regression", "decision_tree", "random_forest"}
    assert set(metrics["model"]) == set(fitted)
    assert {"mae", "rmse", "r2"}.issubset(metrics.columns)
