'''
This script extracts frames from a video file and saves them to a specified directory.

Usage:
    python frame_extractor.py <video_path> <output_dir>
'''
from vid_utils import Vid_Capture
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('output_dir', type=str, default='output_videos', help='Path to the output directory')
    args = parser.parse_args()

    vid_util = Vid_Capture(args.video_path, args.output_dir)
    vid_util.save_frames()
