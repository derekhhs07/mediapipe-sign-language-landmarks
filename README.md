# MediaPipe Sign Language Landmark Classifier

A beginner-friendly real-time sign-language experiment built with Python, MediaPipe, OpenCV and scikit-learn. Instead of classifying raw camera pixels, the program tracks 21 hand landmarks, normalizes their positions and trains an SVM classifier on the resulting finger geometry.

The project has no threshold-image step and no 3D visual effects.

## Features

- Tracks one hand across the full webcam frame
- Uses normalized landmarks and finger-joint angles
- Supports custom static letters or gestures
- Includes an `OTHER` class for incorrect hand shapes
- Rejects low-confidence and ambiguous predictions as `Unknown gesture`
- Uses multiple video frames before displaying a final label
- Optionally includes shoulder, elbow and wrist landmarks for arm gestures

## Setup

Python 3.10 is recommended.

```powershell
python3.10 -m venv .venv-landmarks
.\.venv-landmarks\Scripts\python.exe -m pip install --upgrade pip
.\.venv-landmarks\Scripts\python.exe -m pip install -r requirements_landmarks.txt
```

## Collect training samples

```powershell
.\.venv-landmarks\Scripts\python.exe collect_landmarks.py --label A --samples 50
.\.venv-landmarks\Scripts\python.exe collect_landmarks.py --label OTHER --samples 100
```

In the camera window, press `C` to start or pause collection and `Q` or `Esc` to save and quit. Repeat the command for every label.

## Train

```powershell
.\.venv-landmarks\Scripts\python.exe train_landmark_model.py --confidence 0.70 --margin 0.15 --window 15 --votes 10
```

## Run

```powershell
.\.venv-landmarks\Scripts\python.exe run_landmark_model.py
```

Prediction controls:

- `Q` or `Esc`: quit
- `R`: clear recent predictions
- `+`: increase the confidence requirement
- `-`: decrease the confidence requirement

## Important limitation

This classifier handles static hand shapes. Dynamic ASL letters such as J and Z require fingertip-trajectory tracking across time and are not supported by the current single-frame feature vector.

Captured landmark CSV files and trained model files are intentionally excluded from this repository. Collecting samples from the target webcam and user gives more meaningful real-camera results than using the Sign MNIST 28x28 dataset.

## Acknowledgements

This learning project was inspired by [harshbg/Sign-Language-Interpreter-using-Deep-Learning](https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning) and uses Google's MediaPipe Hand and Pose Landmarker models.

## License

MIT
