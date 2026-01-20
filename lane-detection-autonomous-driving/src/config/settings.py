import numpy as np

# Image Processing Parameters
GAUSSIAN_KERNEL_SIZE = 5
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# ROI Parameters (assuming 1280x720 resolution, can be adjusted)
ROI_VERTICES = np.array([
    [(0, 720), (1280//2 - 50, 450), (1280//2 + 50, 450), (1280, 720)]
], dtype=np.int32)

# Perspective Transform Parameters (Source and Destination points)
# These need calibration, but we'll use approximate values for standard road videos
SRC_POINTS = np.float32([
    [585, 460],  # Top-left
    [203, 720],  # Bottom-left
    [1127, 720], # Bottom-right
    [695, 460]   # Top-right
])

DST_POINTS = np.float32([
    [320, 0],    # Top-left
    [320, 720],  # Bottom-left
    [960, 720],  # Bottom-right
    [960, 0]     # Top-right
])

# Hough Transform Parameters
HOUGH_RHO = 2              # Distance resolution in pixels of the Hough grid
HOUGH_THETA = np.pi/180    # Angular resolution in radians of the Hough grid
HOUGH_THRESHOLD = 15       # Minimum number of votes (intersections in Hough grid cell)
HOUGH_MIN_LINE_LEN = 40    # Minimum number of pixels making up a line
HOUGH_MAX_LINE_GAP = 20    # Maximum gap in pixels between connectable line segments

# Lane Geometry Parameters
LANE_WIDTH_METERS = 3.7    # Standard lane width in meters
XM_PER_PIX = 3.7 / 700     # Meters per pixel in x dimension
YM_PER_PIX = 30 / 720      # Meters per pixel in y dimension

# Paths
VIDEO_INPUT_PATH = 'data/raw/test_video.mp4'
VIDEO_OUTPUT_PATH = 'outputs/videos/test_video_output.mp4'
