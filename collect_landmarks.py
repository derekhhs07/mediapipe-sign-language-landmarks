"""Collect normalized hand (and optional arm) landmark samples from a webcam."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import cv2

from landmark_features import (
    LandmarkTracker,
    camera_frame_closed,
    draw_tracking,
    open_camera,
)


WINDOW_NAME = "Collect landmark gestures"


def parse_arguments() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Collect MediaPipe landmark samples for one gesture label."
    )
    parser.add_argument("--label", required=True, help="Gesture label, for example A or OTHER")
    parser.add_argument("--samples", type=int, default=100, help="Samples for this session")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.12,
        help="Minimum seconds between saved samples",
    )
    parser.add_argument(
        "--use-pose",
        action="store_true",
        help="Also use shoulder, elbow and wrist landmarks for arm-dependent gestures",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (normally selected automatically)",
    )
    arguments = parser.parse_args()
    if arguments.samples < 10:
        parser.error("--samples must be at least 10")
    if arguments.interval < 0.03:
        parser.error("--interval must be at least 0.03 seconds")
    if arguments.output is None:
        filename = "landmark_data_hand_pose.csv" if arguments.use_pose else "landmark_data_hand.csv"
        arguments.output = script_directory / filename
    return arguments


def put_text(
    frame,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.65,
) -> None:
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def prepare_writer(output: Path, feature_count: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_header = ["session", "label", "captured_at", "handedness"] + [
        f"feature_{index:03d}" for index in range(feature_count)
    ]

    if output.exists() and output.stat().st_size:
        with output.open("r", newline="", encoding="utf-8") as existing:
            actual_header = next(csv.reader(existing), [])
        if actual_header != expected_header:
            raise RuntimeError(
                f"{output.name} has a different feature format. Use a new output file "
                "or move the old dataset before collecting."
            )

    handle = output.open("a", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    if output.stat().st_size == 0:
        writer.writerow(expected_header)
        handle.flush()
    return handle, writer


def main() -> None:
    arguments = parse_arguments()
    label = arguments.label.strip().upper()
    if not label or "," in label:
        raise SystemExit("The label must contain visible characters and cannot contain a comma.")

    session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tracker = LandmarkTracker(use_pose=arguments.use_pose)
    data_file, writer = prepare_writer(arguments.output, tracker.feature_count)
    camera = open_camera(arguments.camera)
    if not camera.isOpened():
        tracker.close()
        data_file.close()
        raise SystemExit(
            f"Could not open camera {arguments.camera}. Close other camera apps or try --camera 1."
        )

    captured = 0
    capturing = False
    finished = False
    last_capture = 0.0
    print(f"Collecting label: {label}")
    print("C = start/pause, Q or Esc = save and quit")
    print(f"Output: {arguments.output}")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame could not be read.")
                break

            # A mirrored preview feels natural and the exact same preprocessing is
            # used during both collection and prediction.
            frame = cv2.flip(frame, 1)
            tracking = tracker.process(frame)
            draw_tracking(frame, tracking)

            now = time.monotonic()
            ready = tracking is not None and tracking.features is not None
            if capturing and ready and now - last_capture >= arguments.interval:
                writer.writerow(
                    [
                        session,
                        label,
                        datetime.now().isoformat(timespec="milliseconds"),
                        tracking.handedness,
                        *tracking.features.tolist(),
                    ]
                )
                data_file.flush()
                captured += 1
                last_capture = now
                if captured >= arguments.samples:
                    capturing = False
                    finished = True
                    print(f"Saved {captured} samples for {label}.")

            state_color = (80, 230, 80) if capturing else (0, 210, 255)
            state = "CAPTURING - move hand slightly" if capturing else "PAUSED"
            if finished:
                state = "SESSION COMPLETE - press Q"
                state_color = (80, 230, 80)

            guidance = "No hand - show one hand to the camera"
            if tracking is not None:
                guidance = tracking.guidance

            put_text(frame, f"Label: {label}", (20, 35), (255, 255, 255), 0.8)
            put_text(frame, f"Samples: {captured}/{arguments.samples}", (20, 70))
            put_text(frame, state, (20, 105), state_color)
            put_text(frame, guidance, (20, 140), (80, 230, 80) if ready else (0, 210, 255), 0.58)
            put_text(frame, "C: start/pause   Q or Esc: save and quit", (20, frame.shape[0] - 20), scale=0.55)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c") and not finished:
                capturing = not capturing
                print("Capture started." if capturing else "Capture paused.")
            if camera_frame_closed(WINDOW_NAME):
                break
    finally:
        data_file.close()
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()

    print(f"This session saved {captured} samples for {label} in {arguments.output.name}.")


if __name__ == "__main__":
    main()

