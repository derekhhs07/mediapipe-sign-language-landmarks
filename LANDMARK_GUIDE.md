# Landmark gesture model (no 3D effects)

This is a separate replacement for the old threshold/RGB image pipeline. It
tracks the hand throughout the full camera frame and classifies normalized
landmarks instead of camera pixels.

For static signs such as **A, B, C and 3**, start with hand landmarks only. Use
the optional pose mode later for gestures whose meaning depends on shoulder,
elbow or wrist position.

## 1. Create a clean environment

Open the VS Code PowerShell terminal in this `Code` directory. The old `.venv`
is not required by this new pipeline.

```powershell
python3.10 -m venv .venv-landmarks
.\.venv-landmarks\Scripts\python.exe -m pip install --upgrade pip
.\.venv-landmarks\Scripts\python.exe -m pip install -r requirements_landmarks.txt
```

These commands call the environment's Python directly, so PowerShell script
activation and execution-policy changes are not needed.

If `python3.10` is not recognized, use the Python 3.10 command displayed by
Python Manager, but keep the rest of the first command the same.

## 2. Collect several separate sessions

One command collects 100 samples for one label. A camera window opens; press
`C` to begin, move the hand slightly while holding the correct sign, and press
`Q` or `Esc` to exit. The window also closes safely with its **X** button.

```powershell
.\.venv-landmarks\Scripts\python.exe collect_landmarks.py --label A --samples 100
.\.venv-landmarks\Scripts\python.exe collect_landmarks.py --label B --samples 100
.\.venv-landmarks\Scripts\python.exe collect_landmarks.py --label C --samples 100
.\.venv-landmarks\Scripts\python.exe collect_landmarks.py --label 3 --samples 100
.\.venv-landmarks\Scripts\python.exe collect_landmarks.py --label OTHER --samples 100
```

Run every command three or four times on different occasions. Change distance,
screen position, hand angle and lighting between sessions. For `OTHER`, show
many valid hand shapes that are **not** A, B, C or 3. Do not include an empty
frame: the MediaPipe hand detector already handles "No hand".

All hand-only samples are appended to `landmark_data_hand.csv`. If a session is
poor, delete its rows from the CSV before training or move the whole CSV aside
and recollect it.

The first collection run downloads Google's Hand Landmarker model into the
`mediapipe_models` folder. This is a one-time download.

## 3. Train

```powershell
.\.venv-landmarks\Scripts\python.exe train_landmark_model.py
```

The trainer prints validation accuracy, a per-class report, a confusion matrix,
and the accuracy after uncertain predictions are rejected. It saves
`landmark_gesture_model.joblib`.

If the report says random validation was used, collect at least one additional
separate session for every label and train again. Session-separated validation
is a more realistic measure than mixing neighboring video frames.

## 4. Run the predictor

```powershell
.\.venv-landmarks\Scripts\python.exe run_landmark_model.py
```

Controls:

- `Q` or `Esc`: quit
- `R`: clear the recent-frame prediction history
- `+`: demand higher confidence (fewer but safer predictions)
- `-`: lower the confidence requirement

The large label appears only after the same accepted prediction wins at least
8 of the latest 12 frames. Otherwise the program shows `No hand`, `Hold
steady...`, or `Unknown gesture`.

## Optional: gestures that depend on arm position

Do not mix these samples with the hand-only CSV. Use `--use-pose` during every
collection session:

```powershell
.\.venv-landmarks\Scripts\python.exe collect_landmarks.py --label UP --samples 100 --use-pose
.\.venv-landmarks\Scripts\python.exe collect_landmarks.py --label OTHER --samples 100 --use-pose
.\.venv-landmarks\Scripts\python.exe train_landmark_model.py --data landmark_data_hand_pose.csv --model landmark_gesture_model_pose.joblib
.\.venv-landmarks\Scripts\python.exe run_landmark_model.py --model landmark_gesture_model_pose.joblib
```

Pose mode additionally downloads a lightweight Pose Landmarker model. Keep the
active shoulder, elbow, wrist and hand visible while collecting and predicting.

