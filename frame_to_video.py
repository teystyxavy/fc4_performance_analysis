'''
This script converts frames from a specified directory into a video file using OpenCV.
Usage:
    python frame_to_video.py <frame_path> <output_dir> <vid_path>
'''

from vid_utils import Vid_Capture, Frame_Capture
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frame_path', type=str, help='Path to frames')
    parser.add_argument('output_dir', type=str, default='output_videos', help='Path to the output directory')
    parser.add_argument('vid_path', type=str, default='input_videos/sit_stand.mp4', help='Path to the input video')

    args = parser.parse_args()

    Frame_Capture.convert_frames_to_video(args.frame_path, args.output_dir, args.vid_path)