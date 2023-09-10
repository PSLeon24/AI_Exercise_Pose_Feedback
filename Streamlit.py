import cv2
import streamlit as st
import numpy as np
import mediapipe as mp

st.title("실시간 3대 운동 AI 자세 교정 서비스")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,  # 신뢰도 임계값을 조정
    min_tracking_confidence=0.5,
)

# 신뢰도 임계값을 조절하는 슬라이더 추가
confidence_threshold = st.slider("신뢰도 임계값", 0.0, 1.0, 0.5)

while True:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 포즈 추정
    results_pose = pose.process(frame)

    # 포즈 랜드마크 추출 및 그리기
    if results_pose.pose_landmarks is not None:
        landmarks = results_pose.pose_landmarks.landmark
        # 포즈 랜드마크의 신뢰도를 확인하고 필요한 임계값을 적용
        if (
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility
            >= confidence_threshold
        ):
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

    FRAME_WINDOW.image(frame)
