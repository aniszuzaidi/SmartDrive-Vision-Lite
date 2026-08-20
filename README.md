# Drowsiness Detection System

A real-time drowsiness detection system using EAR (Eye Aspect Ratio) and PERCLOS (Percentage of Eye Closure) with webcam-based eye tracking and yawn detection.

## Features

- **EAR (Eye Aspect Ratio)**: Detects eye open/close states
- **PERCLOS**: Calculates percentage of eye closure over time
- **Yawn Detection**: Detects yawning using mouth aspect ratio
- **Sound Alerts**: Plays alerts when drowsiness is detected
- **Real-time Webcam Detection**: Live video feed analysis

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This project uses MediaPipe instead of dlib for easier installation. MediaPipe works on all platforms without requiring additional build tools.

### 2. (Optional) Add Alert Sound

Place an `alert.wav` file in the project root for custom alert sounds. If not provided, the system will use a beep sound.

## Usage

### Step 1: Train the Model

Train the model on your datasets:

```bash
python train_model.py
```

This will:
- Process images from `Datasets/Open`, `Datasets/Closed`, `Datasets/Yawn`, and `Datasets/NoYawn`
- Extract EAR and MAR features using MediaPipe facial landmarks
- Train Random Forest classifiers
- Save the model as `drowsiness_model.pkl`

**Note**: The training process may take some time depending on the number of images in your dataset.

### Step 2: Run Real-time Detection

Start the eye tracker:

```bash
python eye_tracker.py
```

The system will:
- Open your webcam
- Display real-time detection with:
  - Eye state (Open/Closed)
  - Yawn state (Yawn/No Yawn)
  - EAR value
  - MAR value
  - PERCLOS percentage
- Play sound alerts when drowsiness is detected
- Press 'q' to quit

## Detection Parameters

You can adjust these constants in `eye_tracker.py`:

- `EAR_THRESHOLD = 0.25`: Threshold for eye closure (lower = more sensitive)
- `EAR_CONSECUTIVE_FRAMES = 3`: Frames needed to confirm eye closure
- `MAR_THRESHOLD = 0.5`: Threshold for yawn detection
- `PERCLOS_THRESHOLD = 0.2`: PERCLOS threshold (20% eye closure)
- `FRAME_WINDOW = 30`: Number of frames for PERCLOS calculation

## Dataset Structure

```
Datasets/
├── Open/        (726 images)
├── Closed/      (726 images)
├── Yawn/        (723 images)
└── NoYawn/      (725 images)
```

## Troubleshooting

1. **"No module named 'mediapipe'"**
   - Install MediaPipe: `pip install mediapipe`
   - MediaPipe should install easily on all platforms

2. **Webcam not opening**
   - Check if another application is using the webcam
   - Try changing the camera index in `eye_tracker.py`: `cv2.VideoCapture(1)`

3. **No face detected**
   - Ensure good lighting
   - Face the camera directly
   - Check if face is within frame

4. **Sound alerts not working**
   - Install pygame: `pip install pygame`
   - On Windows, system beep should work automatically

## Model Training

The training script uses:
- **MediaPipe Face Mesh** for facial landmark detection (468 points)
- **Random Forest Classifier** for both eye state and yawn detection
- **10 features**: EAR (left, right, average), MAR, eye dimensions, mouth dimensions
- **80/20 train/test split**

After training, the model accuracy will be displayed, and the model will be saved for use in real-time detection.

**Note**: If some images fail to process (no face detected), they will be skipped. This is normal if images don't contain clear faces.

