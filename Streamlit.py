import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
import datetime
import time
import pygame
import torch

# YOLOv5 모델 불러오기
model_weights_path = "./models/best_big_bounding.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_weights_path)
model.eval()

# 이전 알림 시간 기록
previous_alert_time = 0


# 각도 계산 함수
def calculateAngle(a, b, c):
    a = np.array(a)  # 첫 번째 지점
    b = np.array(b)  # 중간 지점
    c = np.array(c)  # 끝 지점

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# YOLOv5를 사용한 사물 검출 함수
def detect_objects(frame):
    results = model(frame)
    pred = results.pred[0]

    return pred


# Streamlit 앱 초기화
st.title("실시간 3대 운동 AI 자세 교정 서비스")

# Sidebar에 메뉴 추가
menu_selection = st.selectbox("운동 선택", ("벤치프레스", "스쿼트", "데드리프트"))

# Load different models based on the selected exercise
# if menu_selection == "벤치프레스":
#     model_weights_path = './models/benchpress/benchpress.pkl'
# elif menu_selection == "스쿼트":
#     model_weights_path = './models/squat/squat.pkl'
# elif menu_selection == "데드리프트":
#     model_weights_path = './models/deadlift/deadlift.pkl'

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 신뢰도 임계값 조정을 위한 슬라이더 추가
confidence_threshold = st.slider("신뢰도 임계값", 0.0, 1.0, 0.7)

# 각도 표시를 위한 빈 영역 초기화
left_angle_display = st.empty()
right_angle_display = st.empty()

while True:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)

    # YOLOv5를 사용하여 사물 검출
    results_yolo = detect_objects(frame)

    # YOLOv5 결과를 화면에 표시
    if results_yolo is not None:
        for det in results_yolo:
            c1, c2 = det[:2].int(), det[2:4].int()
            cls, conf, *_ = det
            label = f"person {conf:.2f}"

            if conf >= 0.7:  # 신뢰도가 0.7 이상인 경우에만 객체 표시
                # c1과 c2를 튜플로 변환
                c1 = (c1[0].item(), c1[1].item())
                c2 = (c2[0].item(), c2[1].item())

                # YOLOv5로 검출된 객체의 프레임 추출
                object_frame = frame[c1[1] : c2[1], c1[0] : c2[0]]

                # Pose estimation을 수행하기 위해 객체 프레임을 처리
                object_frame_rgb = cv2.cvtColor(object_frame, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(object_frame_rgb)

                # Extract and draw posture landmarks on the object frame
                if results_pose.pose_landmarks is not None:
                    landmarks = results_pose.pose_landmarks.landmark
                    left_hip = [
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                    ]
                    left_knee = [
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
                    ]
                    left_ankle = [
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                    ]
                    right_hip = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
                    ]
                    right_knee = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                    ]
                    right_ankle = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                    ]

                    # 각도 계산
                    left_angle = calculateAngle(left_hip, left_knee, left_ankle)
                    right_angle = calculateAngle(right_hip, right_knee, right_ankle)

                    # 각도 표시 업데이트
                    left_angle_display.text(f"Left Angle: {left_angle:.2f}")
                    right_angle_display.text(f"Right Angle: {right_angle:.2f}")

                    # 화면에 각도 표시
                    frame = cv2.putText(
                        frame,
                        f"Left Angle: {left_angle:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    frame = cv2.putText(
                        frame,
                        f"Right Angle: {right_angle:.2f}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    # 각도 차이 확인
                    angle_diff = abs(left_angle - right_angle)
                    if angle_diff >= 30:
                        current_time = time.time()
                        if current_time - previous_alert_time >= 3:
                            now = datetime.datetime.now()
                            st.write(
                                f"자세가 무너졌습니다! - {now.hour}시 {now.minute}분 {now.second}초"
                            )
                            pygame.mixer.init()
                            pygame.mixer.music.load(
                                "./resources/sounds/broken_posture.mp3"
                            )
                            pygame.mixer.music.play()
                            # 이전 알림 시간 갱신
                            previous_alert_time = current_time

                    # 랜드마크 그리기
                    for landmark in mp_pose.PoseLandmark:
                        if landmarks[landmark.value].visibility >= confidence_threshold:
                            mp.solutions.drawing_utils.draw_landmarks(
                                object_frame,
                                results_pose.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                            )

                # 객체 프레임을 원본 프레임에 다시 그립니다.
                frame[c1[1] : c2[1], c1[0] : c2[0]] = object_frame

                frame = cv2.rectangle(frame, c1, c2, (0, 255, 0), 2)
                frame = cv2.putText(
                    frame,
                    label,
                    (c1[0], c1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    # 원본 프레임을 출력
    FRAME_WINDOW.image(frame)
