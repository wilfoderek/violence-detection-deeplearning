import cv2
import numpy as np

def extract_videos3D(video_input_file_path, height, width):
    video_frames = list()
    cap = cv2.VideoCapture(video_input_file_path)
    while cap.isOpened():

        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (width, height))
            video_frames.append(frame)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return video_frames

def extract_videos3D_optical_flow(video_input_file_path, height, width):
    video_frames_optical_flow = list()
    i = 0
    cap = cv2.VideoCapture(video_input_file_path)
    ret1, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (width, height))
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():

        ret2, frame2 = cap.read()

        if ret2:

            frame2 = cv2.resize(frame2, (width, height))
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            video_frames_optical_flow.append(bgr)
        else:
            break

        i += 1
        prvs = next

    cap.release()
    cv2.destroyAllWindows()
    return video_frames_optical_flow

def extract_videos3D_frames_substraction(video_input_file_path, height, width):
    video_frames = list()
    cap = cv2.VideoCapture(video_input_file_path)
    ret1, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (width, height))

    while cap.isOpened():

        ret2, frame2 = cap.read()
        if ret2:
            frame2 = cv2.resize(frame2, (width, height))
            frame = frame1 - frame2
            video_frames.append(frame)
        else:
            break

        frame1 = frame2

    cap.release()
    cv2.destroyAllWindows()
    return video_frames
