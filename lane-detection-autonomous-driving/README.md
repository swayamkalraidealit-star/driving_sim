# Advanced Lane Detection & Tracking System

This project implements a robust, production-grade lane detection pipeline for autonomous driving simulations. It uses advanced computer vision techniques to detect lane lines, track them smoothly over time, and calculate real-world metrics like vehicle offset and road curvature.

## 🚀 Key Features

*   **Robust Detection**: Uses a combination of **HLS Color Thresholding** (S-channel) and **Sobel Gradient** (X-direction) to detect lines in challenging lighting conditions.
*   **Advanced Lane Modeling**: Replaced basic Hough Lines with a **Second-Order Polynomial Fit** ($x = ay^2 + by + c$) using Sliding Window Search, allowing for accurate curve tracking.
*   **Temporal Smoothing**: Implements **Exponential Moving Average (EMA)** to smooth lane detections across frames, eliminating jitter.
*   **Failure Monitor & Fallback**:
    *   **Sanity Checks**: Validates lane width and parallelism.
    *   **Coast Mode**: "Coasts" on predicted paths if detection is temporarily lost (visualized in Orange).
    *   **Auto-Reset**: Automatically resets tracking logic if lines are lost for too long.
*   **Real-World Metrics**: Computes **Radius of Curvature** and **Vehicle Offset** in meters.
*   **Visual Debugging**: Projects a filled green lane area onto the road surface and displays real-time status/metrics.

## 📂 Project Structure

- `data/`: Input videos (`raw/`) over which the pipeline runs.
- `outputs/`: Processed videos with visualized overlays.
- `src/`: Core source code.
    - `config/`: Centralized settings (`settings.py`) for tuning parameters.
    - `preprocessing/`: resizing, `combined_threshold` logic.
    - `perception/`: `perspective_transform`, `lane_detection` (Polyfit/Sliding Window).
    - `tracking/`: `LaneLine` class for state management and EMA smoothing.
    - `geometry/`: Real-world conversions and curvature math.
    - `visualization/`: Overlay drawing utilities.
    - `pipeline/`: The main `LanePipeline` class integrating all modules.
- `run.py`: Entry point script.

## 🛠️ Installation & Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure**:
    Adjust paths or parameters in `src/config/settings.py` if needed.

3.  **Run the Pipeline**:
    ```bash
    python run.py
    ```
    The output video will be saved to `outputs/videos/`.

## ⚙️ Pipeline Overview

1.  **Input**: Read video frame and resize to 1280x720.
2.  **Thresholding**: Apply HLS S-Channel and Sobel-X thresholds to create a binary map.
3.  **ROI**: Mask the Region of Interest.
4.  **Warp**: Apply Perspective Transform to get a "Bird's-Eye View".
5.  **Detection**:
    -   If tracking: Search around previous polynomial.
    -   If lost/new: Perform full Sliding Window Search.
6.  **Validation**: Check lane width (~3.7m) and parallelism.
7.  **Tracking**: Update `LaneLine` state with EMA smoothing. Handle lost frames.
8.  **Metrics**: Calculate offset from center and curvature radius.
9.  **Visualization**: Unwarp the detected lane area and overlay info on the original frame.

## 🔮 Future Improvements

-   **Kalman Filter**: For even more robust tracking prediction.
-   **Deep Learning**: Replace classical CV with a semantic segmentation network (e.g., U-Net) for complex urban scenes.