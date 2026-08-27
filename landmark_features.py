"""Shared MediaPipe tracking and landmark feature extraction utilities.

This module deliberately contains no gesture classifier.  MediaPipe locates the
hand (and optionally the arm), then the other scripts train a small classifier
on normalized landmark coordinates and joint angles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import os
import time
import urllib.request

import cv2
import numpy as np


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)

HAND_FEATURE_COUNT = 21 * 3 + 15
POSE_FEATURE_COUNT = 10


@dataclass
class TrackingResult:
    features: Optional[np.ndarray]
    hand_landmarks: Sequence
    pose_landmarks: Optional[Sequence]
    handedness: str
    hand_score: float
    usable: bool
    guidance: str


def _download_model(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 100_000:
        return

    partial = destination.with_suffix(destination.suffix + ".download")
    print(f"Downloading MediaPipe model: {destination.name}")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise RuntimeError(
            f"Could not download {destination.name}. Check your internet connection "
            "and run the program again."
        )


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open a camera, preferring DirectShow on Windows to reduce startup delay."""
    if os.name == "nt":
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if camera.isOpened():
            return camera
        camera.release()
    return cv2.VideoCapture(camera_index)


def camera_frame_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


def _angle_cosine(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = a - b
    second = c - b
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-7:
        return 1.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def normalized_hand_features(hand_landmarks: Sequence) -> np.ndarray:
    """Return position-independent hand coordinates plus 15 joint angles."""
    points = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks],
        dtype=np.float32,
    )
    if points.shape != (21, 3):
        raise ValueError(f"Expected 21 hand landmarks, received {points.shape[0]}")

    origin = points[0].copy()
    relative = points - origin

    # The palm-width axis always runs from the pinky MCP to the index MCP.
    # This creates the same local coordinate system at different screen positions.
    x_vector = points[5, :2] - points[17, :2]
    scale = float(np.linalg.norm(x_vector))
    if scale < 1e-5:
        scale = float(np.linalg.norm(points[9, :2] - points[0, :2]))
    scale = max(scale, 1e-5)
    x_axis = x_vector / scale
    y_axis = np.asarray([-x_axis[1], x_axis[0]], dtype=np.float32)
    if float(np.dot(y_axis, points[9, :2] - points[0, :2])) < 0:
        y_axis *= -1.0

    local = np.empty_like(relative)
    local[:, 0] = relative[:, :2] @ x_axis / scale
    local[:, 1] = relative[:, :2] @ y_axis / scale
    local[:, 2] = relative[:, 2] / scale

    angle_features = []
    for chain in FINGER_CHAINS:
        for joint_index in range(1, 4):
            angle_features.append(
                _angle_cosine(
                    local[chain[joint_index - 1]],
                    local[chain[joint_index]],
                    local[chain[joint_index + 1]],
                )
            )

    return np.concatenate(
        (local.reshape(-1), np.asarray(angle_features, dtype=np.float32))
    ).astype(np.float32)


def _body_basis(pose_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    shoulder_vector = pose_points[12, :2] - pose_points[11, :2]
    shoulder_width = max(float(np.linalg.norm(shoulder_vector)), 1e-5)
    x_axis = shoulder_vector / shoulder_width
    y_axis = np.asarray([-x_axis[1], x_axis[0]], dtype=np.float32)
    shoulder_center = (pose_points[11, :2] + pose_points[12, :2]) / 2.0
    hip_center = (pose_points[23, :2] + pose_points[24, :2]) / 2.0
    if float(np.dot(y_axis, hip_center - shoulder_center)) < 0:
        y_axis *= -1.0
    return x_axis, y_axis, shoulder_width


def normalized_arm_features(
    hand_landmarks: Sequence, pose_landmarks: Sequence
) -> np.ndarray:
    """Return active shoulder/elbow/wrist geometry in a body-relative frame."""
    pose = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in pose_landmarks],
        dtype=np.float32,
    )
    hand_wrist = np.asarray(
        [hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z],
        dtype=np.float32,
    )

    # Select the arm whose pose wrist is nearest to the tracked hand wrist.  This
    # avoids relying on left/right labels when using a mirrored webcam image.
    arm_indices = min(
        ((11, 13, 15), (12, 14, 16)),
        key=lambda arm: np.linalg.norm(pose[arm[2], :2] - hand_wrist[:2]),
    )
    shoulder_index, elbow_index, wrist_index = arm_indices
    x_axis, y_axis, scale = _body_basis(pose)
    shoulder = pose[shoulder_index]

    def convert(point: np.ndarray) -> list[float]:
        relative = point - shoulder
        return [
            float(np.dot(relative[:2], x_axis) / scale),
            float(np.dot(relative[:2], y_axis) / scale),
            float(relative[2] / scale),
        ]

    elbow = convert(pose[elbow_index])
    wrist = convert(pose[wrist_index])
    detected_hand = convert(hand_wrist)
    elbow_angle = _angle_cosine(
        pose[shoulder_index], pose[elbow_index], pose[wrist_index]
    )
    return np.asarray(elbow + wrist + detected_hand + [elbow_angle], dtype=np.float32)


def validate_hand(hand_landmarks: Sequence) -> tuple[bool, str]:
    points = np.asarray([[point.x, point.y] for point in hand_landmarks])
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    width, height = maximum - minimum

    if minimum[0] < 0.015 or minimum[1] < 0.015 or maximum[0] > 0.985 or maximum[1] > 0.985:
        return False, "Move the entire hand into the camera"
    if max(width, height) < 0.10:
        return False, "Move your hand closer"
    if max(width, height) > 0.92:
        return False, "Move your hand slightly farther away"
    return True, "Hand ready"


class LandmarkTracker:
    def __init__(
        self,
        use_pose: bool = False,
        model_directory: Optional[Path] = None,
        detection_confidence: float = 0.70,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as error:
            raise RuntimeError(
                "MediaPipe is not installed. Follow LANDMARK_GUIDE.md to create "
                "the .venv-landmarks environment."
            ) from error

        self.mp = mp
        self.vision = vision
        self.use_pose = use_pose
        models = model_directory or Path(__file__).resolve().parent / "mediapipe_models"
        hand_model = models / "hand_landmarker.task"
        _download_model(HAND_MODEL_URL, hand_model)

        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=detection_confidence,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        self.pose_detector = None

        if use_pose:
            pose_model = models / "pose_landmarker_lite.task"
            _download_model(POSE_MODEL_URL, pose_model)
            pose_options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(pose_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.60,
                min_pose_presence_confidence=0.60,
                min_tracking_confidence=0.60,
                output_segmentation_masks=False,
            )
            self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        self._last_timestamp = 0

    @property
    def feature_count(self) -> int:
        return HAND_FEATURE_COUNT + (POSE_FEATURE_COUNT if self.use_pose else 0)

    def process(self, frame_bgr: np.ndarray) -> Optional[TrackingResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(int(time.monotonic() * 1000), self._last_timestamp + 1)
        self._last_timestamp = timestamp

        hand_result = self.hand_detector.detect_for_video(image, timestamp)
        if not hand_result.hand_landmarks:
            return None

        hand_landmarks = hand_result.hand_landmarks[0]
        handedness = "Unknown"
        hand_score = 0.0
        if hand_result.handedness and hand_result.handedness[0]:
            category = hand_result.handedness[0][0]
            handedness = category.category_name or category.display_name or "Unknown"
            hand_score = float(category.score)

        usable, guidance = validate_hand(hand_landmarks)
        pose_landmarks = None
        pose_features = None
        if self.pose_detector is not None:
            pose_result = self.pose_detector.detect_for_video(image, timestamp)
            if pose_result.pose_landmarks:
                pose_landmarks = pose_result.pose_landmarks[0]
                pose_features = normalized_arm_features(hand_landmarks, pose_landmarks)
            else:
                usable = False
                guidance = "Move back until shoulder, elbow and wrist are visible"

        features = None
        if usable:
            features = normalized_hand_features(hand_landmarks)
            if self.use_pose:
                if pose_features is None:
                    features = None
                else:
                    features = np.concatenate((features, pose_features)).astype(np.float32)

        return TrackingResult(
            features=features,
            hand_landmarks=hand_landmarks,
            pose_landmarks=pose_landmarks,
            handedness=handedness,
            hand_score=hand_score,
            usable=usable,
            guidance=guidance,
        )

    def close(self) -> None:
        self.hand_detector.close()
        if self.pose_detector is not None:
            self.pose_detector.close()

    def __enter__(self) -> "LandmarkTracker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def draw_tracking(frame: np.ndarray, tracking: Optional[TrackingResult]) -> None:
    if tracking is None:
        return

    height, width = frame.shape[:2]
    hand_points = [
        (int(point.x * width), int(point.y * height))
        for point in tracking.hand_landmarks
    ]
    color = (70, 220, 70) if tracking.usable else (0, 190, 255)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, hand_points[start], hand_points[end], color, 2, cv2.LINE_AA)
    for point in hand_points:
        cv2.circle(frame, point, 4, (30, 30, 255), -1, cv2.LINE_AA)

    if tracking.pose_landmarks is not None:
        pose_points = [
            (int(point.x * width), int(point.y * height))
            for point in tracking.pose_landmarks
        ]
        for shoulder, elbow, wrist in ((11, 13, 15), (12, 14, 16)):
            cv2.line(frame, pose_points[shoulder], pose_points[elbow], (255, 170, 30), 3)
            cv2.line(frame, pose_points[elbow], pose_points[wrist], (255, 170, 30), 3)
            for index in (shoulder, elbow, wrist):
                cv2.circle(frame, pose_points[index], 5, (255, 100, 20), -1)

