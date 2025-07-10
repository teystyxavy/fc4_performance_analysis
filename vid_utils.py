import cv2
import os
import re

class Vid_Capture:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.output_path = os.path.join(output_dir, os.path.basename(video_path))
        try:
            self.cap = cv2.VideoCapture(video_path)
        except:
            print('Error opening video file:', video_path)
            self.cap.release()
            return
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps, self.width, self.height, self.total_frames = self.get_video_properties(self.cap)
        self.video_writer = self.initialize_video_writer()
        

    def get_video_properties(self, cap):
        # get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return original_fps, width, height, total_frames
    
    def initialize_video_writer(self):
        video_writer = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

        if not video_writer.isOpened():
            print('Error opening video writer')
            return
        
        return video_writer
    
    def resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        '''resize image with aspect ratio'''    
        # get the current width and height
        w, h = image.shape[1], image.shape[0]
        
        # calculate the scaling factor
        if width is None and height is None:
            return image
        if width is None:   
            scale = height / h
            dim = (int(w * scale), height)
        else:
            scale = width / w
            dim = (width, int(h * scale))

        return cv2.resize(image, dim, interpolation=inter)
    
    def save_frames(self):
        os.makedirs(self.output_dir, exist_ok=True)
        processed_count = 0
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret == False:
                    break

                processed_count += 1
                cv2.imwrite(os.path.join(self.output_dir, f'{processed_count}_{os.path.basename(self.video_path).split(".")[0]}.jpg'), frame)
                if processed_count % 50 == 0:
                    print(f'Processed {processed_count} frames / {self.total_frames} frames')

            print('Video processing completed, all frames saved to:', self.output_path)

        except Exception as e:
            e
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

class Frame_Capture:

    # CRITICAL: Proper numerical sorting to prevent frame order issues
    def natural_sort_key(filename):
        # Extract numbers from filename for proper sorting
        numbers = re.findall(r'\d+', filename)
        return [int(num) for num in numbers] if numbers else [0]
        
    
    @staticmethod
    def get_video_properties(cap):
        # get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return original_fps, width, height, total_frames

    @staticmethod
    def initialize_video_writer(output_path, fourcc, fps, width, height):
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not video_writer.isOpened() or video_writer is None:
            print('Error opening video writer')
            return
        
        return video_writer
    
    @staticmethod
    def convert_frames_to_video(frame_dir, output_dir, vid_path):
        os.makedirs(output_dir, exist_ok=True)

        frame_paths = [os.path.join(frame_dir, frame_name) for frame_name in (os.listdir(frame_dir))]
        frame_paths.sort(key=Frame_Capture.natural_sort_key)
        total_frames= len(frame_paths)

        # take first frame as reference to get video properties
        height, width = cv2.imread(frame_paths[0]).shape[:2]

        output_path = os.path.join(output_dir, 'corrected_' + os.path.basename(vid_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        print('Initializing video writer')
        video_writer = Frame_Capture.initialize_video_writer(output_path, fourcc, fps, width, height)

        processed_count = 0
        
        try:
            print('Writing frames to video')
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                video_writer.write(frame)

                processed_count += 1
                if processed_count % 500 == 0:
                    print(f'Processed {processed_count} frames / {total_frames} frames')

        finally:
            print('Video processing completed, all frames saved to:', output_dir)
            print('Total frames:', total_frames)
            video_writer.release()