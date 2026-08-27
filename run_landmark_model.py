"""Run confidence-filtered, temporally smoothed webcam gesture prediction."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from landmark_features import (
    LandmarkTracker,
    camera_frame_closed,
    draw_tracking,
    open_camera,
)


WINDOW_NAME = "Landmark sign-language predictor"


def parse_arguments() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run a trained landmark gesture model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=script_directory / "landmark_gesture_model.joblib",
    )
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def put_text(
    frame,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.7,
    thickness: int = 2,
) -> None:
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def stable_prediction(history: deque, required_votes: int):
    visible = [label for label in history if label is not None]
    if not visible:
        return None
    label, count = Counter(visible).most_common(1)[0]
    if count >= required_votes:
        return label
    return None


def main() -> None:
    arguments = parse_arguments()
    if not arguments.model.exists():
        raise SystemExit(
            f"Model not found: {arguments.model}\nRun train_landmark_model.py first."
        )

    bundle = joblib.load(arguments.model)
    required_keys = {
        "pipeline",
        "feature_count",
        "use_pose",
        "confidence_threshold",
        "margin_threshold",
        "smoothing_window",
        "required_votes",
    }
    missing = required_keys.difference(bundle)
    if missing:
        raise SystemExit(f"The model bundle is missing fields: {sorted(missing)}")

    classifier = bundle["pipeline"]
    confidence_threshold = float(bundle["confidence_threshold"])
    margin_threshold = float(bundle["margin_threshold"])
    history = deque(maxlen=int(bundle["smoothing_window"]))
    required_votes = int(bundle["required_votes"])

    tracker = LandmarkTracker(use_pose=bool(bundle["use_pose"]))
    if tracker.feature_count != int(bundle["feature_count"]):
        tracker.close()
        raise SystemExit("The tracker feature format does not match the trained model.")

    camera = open_camera(arguments.camera)
    if not camera.isOpened():
        tracker.close()
        raise SystemExit(
            f"Could not open camera {arguments.camera}. Close other camera apps or try --camera 1."
        )

    print("Q or Esc = quit, R = clear prediction, + / - = adjust confidence")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame could not be read.")
                break

            frame = cv2.flip(frame, 1)
            tracking = tracker.process(frame)
            draw_tracking(frame, tracking)

            final_label = None
            raw_label = "-"
            raw_confidence = 0.0
            raw_margin = 0.0
            status = "No hand"
            status_color = (0, 210, 255)

            if tracking is None:
                history.clear()
            elif tracking.features is None:
                history.append(None)
                status = tracking.guidance
            else:
                probabilities = classifier.predict_proba(tracking.features.reshape(1, -1))[0]
                order = np.argsort(probabilities)
                best_index = int(order[-1])
                second_index = int(order[-2])
                raw_label = str(classifier.classes_[best_index])
                raw_confidence = float(probabilities[best_index])
                raw_margin = raw_confidence - float(probabilities[second_index])

                candidate = None
                if (
                    raw_label != "OTHER"
                    and raw_confidence >= confidence_threshold
                    and raw_margin >= margin_threshold
                ):
                    candidate = raw_label
                history.append(candidate)
                final_label = stable_prediction(history, required_votes)

                if final_label is not None:
                    status = final_label
                    status_color = (70, 235, 70)
                elif candidate is not None:
                    status = "Hold steady..."
                else:
                    status = "Unknown gesture"

            put_text(frame, status, (20, 55), status_color, 1.15, 3)
            put_text(
                frame,
                f"Raw: {raw_label}  confidence: {raw_confidence:.2f}  margin: {raw_margin:.2f}",
                (20, 92),
                (230, 230, 230),
                0.55,
            )
            put_text(
                frame,
                f"Required: confidence {confidence_threshold:.2f}, margin {margin_threshold:.2f}",
                (20, 120),
                (230, 230, 230),
                0.52,
            )
            put_text(
                frame,
                "Q/Esc: quit   R: reset   +/-: change confidence",
                (20, frame.shape[0] - 20),
                (255, 255, 255),
                0.52,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                history.clear()
            if key in (ord("+"), ord("=")):
                confidence_threshold = min(0.98, confidence_threshold + 0.05)
                history.clear()
            if key in (ord("-"), ord("_")):
                confidence_threshold = max(0.50, confidence_threshold - 0.05)
                history.clear()
            if camera_frame_closed(WINDOW_NAME):
                break
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

