import cv2
from moviepy import VideoFileClip
from src.pipeline.lane_pipeline import LanePipeline
from src.config import settings

def main():
    pipeline = LanePipeline()

    # Load video
    print(f"Processing video: {settings.VIDEO_INPUT_PATH}")
    clip = VideoFileClip(settings.VIDEO_INPUT_PATH)
    
    # Process video
    output_clip = clip.image_transform(pipeline.process_frame)
    
    # Write output
    print(f"Saving output to: {settings.VIDEO_OUTPUT_PATH}")
    output_clip.write_videofile(settings.VIDEO_OUTPUT_PATH, audio=False)

if __name__ == "__main__":
    main()
