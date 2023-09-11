import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
import datetime
import time  # 시간 모듈 추가
import pygame

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


st.title("실시간 3대 운동 AI 자세 교정 서비스")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5  # 신뢰도 임계값 조정
)

# 신뢰도 임계값 조정을 위한 슬라이더 추가
confidence_threshold = st.slider("신뢰도 임계값", 0.0, 1.0, 0.5)

while True:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose estimation
    results_pose = pose.process(frame)

    # Extract and draw posture landmarks
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
                st.write(f"자세가 무너졌습니다! - {now.hour}시 {now.minute}분 {now.second}초")
                pygame.mixer.init()
                pygame.mixer.music.load("broken_posture.mp3")
                pygame.mixer.music.play()
                # 'broken_posture.mp3' 오디오 파일을 HTML <audio> 요소로 표시
                # audio_file = "broken_posture.mp3"
                # audio_html = f'<audio src="data:audio/mp3;base64,{base64.b64encode(open(audio_file, "rb").read()).decode()}" controls autoplay></audio>'
                # st.markdown(audio_html, unsafe_allow_html=True)
                # 이전 알림 시간 갱신
                previous_alert_time = current_time

        # 운동 자세 그래픽 표시
        for landmark in mp_pose.PoseLandmark:
            if landmarks[landmark.value].visibility >= confidence_threshold:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                )

    FRAME_WINDOW.image(frame)
