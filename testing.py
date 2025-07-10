from vid_utils import Vid_Capture
import cv2 
import numpy as np
import argparse
import os
from datetime import datetime
import pandas as pd
from decimal import Decimal, getcontext

getcontext().prec = 10

def generate_results_pose(model, video_path, output_path, total_frames, 
                          processed_count, total_landmarks, total_metric, 
                          conf_dict):
    metric = 'visibility' if model == 'mediapipe' else 'confidence'

    avg_conf_dict = {name: v/total_frames for name, v in conf_dict.items()}
    results_string = '----------------------------------------\n'
    results_string += f'Input video: {video_path}\n'
    results_string += f'Output video: {output_path}\n'
    results_string += f'Input frames: {total_frames}\n'
    results_string += f'Processed frames: {processed_count}\n'
    results_string += f'Total landmarks: {total_landmarks}\n'
    results_string += f'Total {metric}: {total_metric:.5f}\n'
    results_string += f'Total {metric} per landmark: {[f'{name} : {conf:.5f}' for name, conf in conf_dict.items()]}\n'
    results_string += f'Average landmarks per frame: {(total_landmarks / processed_count):.5f}\n'
    results_string += f'Average {metric} per frame: {(total_metric/ processed_count):.5f}\n'
    results_string += f'Average {metric} per landmark: {(total_metric / total_landmarks):.5f}\n'
    results_string += f'Average {metric} per landmark: {[f'{name} : {avg_conf:.5f}' for name, avg_conf in avg_conf_dict.items()]}\n'

    return results_string

def generate_results_object(input_path, output_path, total_frames, 
                          processed_count, total_objects, total_confidence, 
                          conf_dict, count_dict):
    
    avg_conf_dict = {name: float(v/count_dict[name]) for name, v in conf_dict.items()}
    avg_conf_per_frame = float(total_confidence / processed_count) if processed_count > 0 else 0
    avg_conf_per_object = float(total_confidence / total_objects) if total_objects > 0 else 0

    results_string = '----------------------------------------\n'
    results_string += f'Input video: {input_path}\n'
    results_string += f'Output video: {output_path}\n'
    results_string += f'Input frames: {total_frames}\n'
    results_string += f'Processed frames: {processed_count}\n'
    results_string += f'Total objects: {total_objects}\n'
    results_string += f'Total confidence: {total_confidence:.5f}\n'
    results_string += f'Total confidence per object: {[f'{name} : {conf:.5f}' for name, conf in conf_dict.items()]}\n'
    results_string += f'Average confidence per frame: {avg_conf_per_frame:.5f}\n'
    results_string += f'Average confidence per object: {avg_conf_per_object:.5f}\n'
    results_string += 'Class-wise Information:\n'
    results_string += f'No. of class instances: {count_dict}\n'
    results_string += f'Average confidence for each class: {[f"{name}: {avg_conf:.5f}" for name, avg_conf in avg_conf_dict.items()]}\n'

    return results_string



def check_gpu():
    import torch
    print('GPU Available:', torch.cuda.is_available())
    print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU found')
    print('Number of GPUs:', torch.cuda.device_count())

# create processing pipeline
def pose_est_mediapipe(video_path, output_dir, display_frame=False):

    import mediapipe as mp
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # global variables
    model_path = 'models/pose_landmarker_full.task'

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options = BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
    )

    os.makedirs(output_dir, exist_ok=True)

    vid_util = Vid_Capture(video_path, output_dir)
    cap = vid_util.cap
    video_writer = vid_util.video_writer # to write video

    # get video properties
    original_fps, width, height, total_frames = vid_util.get_video_properties(cap)
    processed_count = 0
    total_landmarks = 0
    total_visibility = 0

    idx_to_class = {
    0:  'nose',
    1:  'left_eye_inner',
    2:  'left_eye',
    3:  'left_eye_outer',
    4:  'right_eye_inner',
    5:  'right_eye',
    6:  'right_eye_outer',
    7:  'left_ear',
    8:  'right_ear',
    9:  'mouth_left',
    10: 'mouth_right',
    11: 'left_shoulder',
    12: 'right_shoulder',
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
    17: 'left_pinky',
    18: 'right_pinky',
    19: 'left_index',
    20: 'right_index',
    21: 'left_thumb',
    22: 'right_thumb',
    23: 'left_hip',
    24: 'right_hip',
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle',
    29: 'left_heel',
    30: 'right_heel',
    31: 'left_foot_index',
    32: 'right_foot_index'
    }

    conf_dict = {name:0 for name in idx_to_class.values()}

    with PoseLandmarker.create_from_options(options) as landmarker:
        try:
            while cap.isOpened():
                    ret, frame = cap.read()
                    if ret == False:
                        break
                
                    frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                    pose_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_timestamp_ms))
                    pose_landmarks_list = pose_landmarker_result.pose_landmarks
                    annotated_image = np.copy(frame)

                    for i, pose_landmarks in enumerate(pose_landmarks_list):

                        # draw landmarks
                        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                        to_extend = [
                            landmark_pb2.NormalizedLandmark(
                                x=landmark.x, 
                                y=landmark.y, 
                                z=landmark.z
                            ) for landmark in pose_landmarks
                        ]
                        pose_landmarks_proto.landmark.extend(to_extend)

                        total_landmarks += len(to_extend)

                        for i, landmark in enumerate(pose_landmarks):
                            vis = landmark.visibility
                            conf_dict[idx_to_class[i]] += vis
                            total_visibility += vis
                        
                        solutions.drawing_utils.draw_landmarks(
                            annotated_image,
                            pose_landmarks_proto,
                            solutions.pose.POSE_CONNECTIONS,
                            solutions.drawing_styles.get_default_pose_landmarks_style()
                        )

                        # save annotated image
                        video_writer.write(annotated_image)
                        processed_count += 1

                        if processed_count % 100 == 0:
                            print(f'Processed {processed_count} frames / {total_frames} frames')
                            print(f'Total landmarks: {total_landmarks}')
                            print(f'Total visibility: {total_visibility}')

                        if display_frame:
                            final_frame = vid_util.resize_with_aspect_ratio(annotated_image, width=1000, height=1000)

                            cv2.imshow('MediaPipe Pose Estimation', final_frame)
                            key = cv2.waitKey(1) & 0xFF

                            if key == ord('q'):
                                print('Processing stopped by user')
                                break
                            elif key == ord('p'):
                                print('Processing paused, press any key to resume')
                                cv2.waitKey(0)
                                print('Processing resumed')
                
        except Exception as e:
            print(e)
        finally:
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()
            results_string = generate_results_pose(model='mediapipe',
                                                   video_path=video_path,
                                                   output_path=vid_util.output_path,
                                                   total_frames=total_frames,
                                                   processed_count=processed_count,
                                                   total_landmarks=total_landmarks,
                                                   total_metric=total_visibility,
                                                   conf_dict=conf_dict)

            print('Video processing completed')
            print(results_string)

            with open('log_results.txt', 'a') as f:
                f.write(results_string)

def pose_est_yolo(video_path, output_dir, display_frame=False):
    from ultralytics import YOLO
    
    os.makedirs(output_dir, exist_ok=True)

    vid_util = Vid_Capture(video_path, output_dir)
    cap = vid_util.cap
    video_writer = vid_util.video_writer # to write video

    # get video properties
    original_fps, width, height, total_frames = vid_util.get_video_properties(cap)
    processed_count = 0
    total_landmarks = 0
    total_persons = 0
    total_confidence = 0

    idx_to_class = {
        0:'nose', 
        1:'left_eye', 
        2:'right_eye', 
        3:'left_ear', 
        4:'right_ear', 
        5:'left_shoulder', 
        6:'right_shoulder', 
        7:'left_elbow', 
        8:'right_elbow', 
        9:'left_wrist', 
        10:'right_wrist', 
        11:'left_hip', 
        12:'right_hip', 
        13:'left_knee', 
        14:'right_knee', 
        15:'left_ankle', 
        16:'right_ankle'
    }
    
    conf_dict = {name:0 for name in idx_to_class.values()}

    pose_model = YOLO('yolo11l-pose.pt')

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break

            results = pose_model(frame, device=[0])

            result = results[0]
            annotated_frame = result.plot(img=frame)

            if result.boxes is not None and result.boxes.data is not None:
                boxes = result.boxes
                total_persons += len(boxes)

            if result.keypoints is not None and result.keypoints.data is not None:
                keypoints = result.keypoints

                total_persons += len(keypoints)

                if hasattr(keypoints, 'data') and keypoints.data is not None:
                    for keypoints_data in keypoints.data:
                        num_keypoints, _ = keypoints_data.shape
                        total_landmarks += num_keypoints

                for conf in keypoints.conf:
                    total_confidence += conf.cpu().numpy().sum()
                    for i, indiv_conf in enumerate(conf.cpu().numpy()):
                        conf_dict[idx_to_class[i]] += float(indiv_conf)

            # save annotated image
            video_writer.write(annotated_frame)
            processed_count += 1

            if processed_count % 100 == 0:
                print(f'Processed {processed_count} frames / {total_frames} frames')
                print(f'Total landmarks: {total_landmarks}')
                print(f'Total visibility: {total_confidence}')

            if display_frame:
                final_frame = vid_util.resize_with_aspect_ratio(annotated_frame, width=1000, height=1000)

                cv2.imshow('YOLO Pose Estimation', final_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print('Processing stopped by user')
                    break
                elif key == ord('p'):
                    print('Processing paused, press any key to resume')
                    cv2.waitKey(0)
                    print('Processing resumed')
                
    except Exception as e:
        print(e)
    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        avg_conf_dict = {name: v/total_persons for name, v in conf_dict.items()}

        results_string = generate_results_pose(model='yolo', 
                                               video_path=video_path,
                                               output_path=vid_util.output_path,
                                               total_frames=total_frames,
                                               processed_count=processed_count,
                                               total_landmarks=total_landmarks,
                                               total_metric=total_confidence,
                                               conf_dict=conf_dict)

        print('Video processing completed')
        print(results_string)

        with open('log_results.txt', 'a') as f:
            f.write(results_string)

def object_detection_yolo(video_path, output_dir, model_path, display_frame=False):
    from ultralytics import YOLO

    os.makedirs(output_dir, exist_ok=True)

    vid_util = Vid_Capture(video_path, output_dir)
    cap = vid_util.cap
    video_writer = vid_util.video_writer # to write video

    # get video properties
    original_fps, width, height, total_frames = vid_util.get_video_properties(cap)
    processed_count = 0
    total_objects = 0
    total_confidence = 0

    conf_dict = {}

    count_dict = {}

    det_model = YOLO(model_path)
    
    results = det_model(video_path, stream=True, verbose=False)

    for result in results:
        annotated_frame = result.plot()

        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for cls, conf in zip(classes, confidences):
                name = result.names[int(cls)]

                if name not in conf_dict.keys():
                    conf_dict[name] = 0
                if name not in count_dict.keys():
                    count_dict[name] = 0
                conf_dict[name] += conf
                total_confidence += conf
                count_dict[name] += 1
                total_objects += 1

        # save annotated image
        video_writer.write(annotated_frame)
        processed_count += 1

        if processed_count % 100 == 0:
                print(f'Processed {processed_count} frames / {total_frames} frames')
                print(f'Total objects: {total_objects}')
                print(f'Total confidence: {total_confidence}')

        if display_frame:
            final_frame = vid_util.resize_with_aspect_ratio(annotated_frame, width=1000, height=1000)

            cv2.imshow('YOLO Pose Estimation', final_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print('Processing stopped by user')
                break
            elif key == ord('p'):
                print('Processing paused, press any key to resume')
                cv2.waitKey(0)
                print('Processing resumed')
            
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    results_string = generate_results_object(input_path=video_path,
                                              output_path=vid_util.output_path,
                                              total_frames=total_frames,
                                              processed_count=processed_count,
                                              total_objects=total_objects,
                                              total_confidence=float(total_confidence),
                                              conf_dict=conf_dict,
                                              count_dict=count_dict)
    print('Video processing completed')
    print(results_string)

    with open('log_results.txt', 'a') as f:
        f.write(results_string)

def image_object_detection_yolo(folder_path, output_dir, model_path, display_frame=False):
    from ultralytics import YOLO

    os.makedirs(output_dir, exist_ok=True)
    det_model = YOLO(model_path)

    conf_dict = {}
    image_conf_dict = {}
    count_dict = {}

    image_paths = os.listdir(folder_path)
    image_paths = [os.path.join(folder_path, image_path) for image_path in image_paths]
    total_images = len(image_paths)

    processed_count = 0
    total_objects = 0
    total_confidence = 0

    results = det_model(image_paths, stream=True, verbose=False, device=[0])

    for i, result in enumerate(results):
        annotated_frame = result.plot()

        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for cls, conf in zip(classes, confidences):
                name = result.names[int(cls)]

                total_confidence += float(conf)
                if name not in conf_dict.keys():
                    conf_dict[name] = 0
                conf_dict[name] += float(conf)
                curr_list = image_conf_dict.get(image_paths[i], [])
                curr_list.append((name, float(conf)))
                image_conf_dict[image_paths[i]] = curr_list
                count_dict[name] = count_dict.get(name, 0) + 1
                total_objects += 1
            
        # save annotated image
        output_path = os.path.join(output_dir, os.path.basename(image_paths[i]))
        cv2.imwrite(output_path, annotated_frame)
        processed_count += 1

        
        print(f'Processed {processed_count} images / {len(image_paths)} images')
        print(f'Total objects: {total_objects}')
        print(f'Total confidence: {total_confidence}')

    results_string = generate_results_object(input_path=folder_path,
                                              output_path=output_dir,
                                              total_frames=total_images,
                                              processed_count=processed_count,
                                              total_objects=total_objects,
                                              total_confidence=float(total_confidence),
                                              conf_dict=conf_dict,
                                              count_dict=count_dict)
    results_string += 'Detections and confidence for each image:'
    results_string += str([
        f"{os.path.basename(path)}: ({name}, {conf:.5f})"
        for path, detections in image_conf_dict.items()
        for name, conf in detections
        ]) + "\n"
    
    print(results_string)

    with open('log_results.txt', 'a') as f:
        f.write(results_string)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='Path to the video or images folder')
    parser.add_argument('-o', '--output_dir', type=str, default='output_vids', help='Path to the output directory')
    parser.add_argument('-m', '--model_type', type=str, default='mediapose', help='Type of model to use')
    parser.add_argument('-d', '--display_frame', action='store_true', default=False, help='Display the processed frame')
    args = parser.parse_args()

    if args.model_type != 'mediapose':
        check_gpu()
            

    if args.path.endswith('.mp4'):
        if args.model_type == 'mediapose':
            pose_est_mediapipe(args.path, os.path.join(args.output_dir, 'mediapose'), args.display_frame)
        elif args.model_type == 'yolo-pose':
            pose_est_yolo(args.path, os.path.join(args.output_dir, 'yolo', 'pose'), args.display_frame)
        elif args.model_type == 'yolo-object':
            object_detection_yolo(args.path, os.path.join(args.output_dir, 'yolo', 'object'), 'models/yolo11l.pt', args.display_frame)
    
    else: # path is image folder
        if args.model_type == 'yolo-pill':
            model_path = 'av_med_16-12-24.pt'
        else:
            model_path = 'yolo11l.pt'
        image_object_detection_yolo(args.path, os.path.join(args.output_dir, 'yolo', 'object'), os.path.join('models', model_path))

    


