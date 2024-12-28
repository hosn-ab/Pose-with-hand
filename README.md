# YOLO Pose Detection with Hand Position Integration

This project demonstrates how to enhance YOLO-based pose detection by incorporating hand position detection. The integration allows for more robust and precise pose detection with added hand-specific keypoints.

## Features

- **YOLO Pose Detection**: Detects human poses efficiently.
- **Hand Position Integration**: Adds hand detection, improving pose estimation accuracy by incorporating hand-specific keypoints.
- **Visualization**: Displays detected poses with hand positions and confidence levels overlayed on the input frames.

## Prerequisites

1. Python 3.8 or later
2. Required Python libraries:
   - `ultralytics`
   - `opencv-python`
   - `numpy`
   - `imutils`

Install the dependencies using:
```bash
pip install ultralytics opencv-python numpy imutils
```

3. Trained models:
   - `hand.pt` for hand detection
   - `yolo11n-pose.pt` for pose detection
   - Download these models and place them in the project directory.

## Project Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Ensure the required models (`hand.pt` and `yolo11n-pose.pt`) are in the working directory.

## Running the Code

1. Connect a webcam or provide a video input stream.
2. Run the main script:
   ```bash
   python inference.py
   ```
3. The program starts capturing frames, detecting poses, and adding hand positions.

4. Key functionalities:
   - Press `q` to quit the application.

## How It Works

### Enhancing YOLO Pose Detection
The script builds upon YOLO's pose detection capabilities by adding hand keypoints as additional markers. This is achieved through the following steps:

1. **Hand Detection**:
   - Using a fine-tuned YOLO model (`hand.pt`), hand positions are detected from input frames.

2. **Pose Extension**:
   - Detected hand positions are matched to existing wrist keypoints from the pose detection model (`yolo11n-pose.pt`).
   - The wrist keypoints are extended to include additional hand markers (index 17 and 18 in the keypoints array).

3. **Visualization**:
   - Detected poses with hands are drawn on the frame, including confidence scores for hand detection.

### Key Functions
- `adjust_hand_data`: Prepares hand detection results for integration.
- `match_hand`: Matches detected hands with pose keypoints, extending pose keypoints with hand data.
- `plot_poses`: Visualizes the poses with extended hand keypoints and their confidence scores.

## Note
I am making this project public for anyone who might benefit from this model and approach. If you have any feedback or questions, feel free to reach out.
