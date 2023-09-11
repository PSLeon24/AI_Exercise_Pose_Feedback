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
    min_detection_confidence=0.5, min_tracking_confidence=0.5  # 신뢰도 임계값 조정
)

# 신뢰도 임계값 조정을 위한 슬라이더 추가
confidence_threshold = st.slider("신뢰도 임계값", 0.0, 1.0, 0.5)

while True:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 자세 추정
    results_pose = pose.process(frame)

    # 자세 랜드마크 추출 및 그리기
    if results_pose.pose_landmarks is not None:
        landmarks = results_pose.pose_landmarks.landmark
        # 모든 관절에 대한 신뢰도를 확인하고 조건을 적용하여 자세를 그래픽으로 표시
        for landmark in mp_pose.PoseLandmark:
            if landmarks[landmark.value].visibility >= confidence_threshold:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                )

    FRAME_WINDOW.image(frame)
