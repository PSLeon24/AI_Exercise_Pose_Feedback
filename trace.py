import cv2
import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
import torch

st.set_page_config(
    page_title="이미지 관절 지점 및 YOLOv5 객체 검출",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# YOLOv5 모델 불러오기
model_weights_path = "./models/best_big_bounding.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_weights_path)
model.to("mps")
model.eval()

# 이미지 불러오기
image_path = "bench2.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# YOLOv5를 사용하여 사물 검출 함수
def detect_objects(frame):
    results = model(frame)
    pred = results.pred[0]
    return pred


# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.3, min_tracking_confidence=0.7, model_complexity=2
)

# 이미지에서 객체 검출
results_yolo = detect_objects(image)

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
            object_frame = image[c1[1] : c2[1], c1[0] : c2[0]]

            # Pose estimation을 수행하기 위해 객체 프레임을 처리
            object_frame_rgb = cv2.cvtColor(object_frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(object_frame_rgb)

            if results_pose.pose_landmarks is not None:
                landmarks = results_pose.pose_landmarks.landmark

                # 랜드마크 표시
                for landmark in mp_pose.PoseLandmark:
                    if landmarks[landmark.value].visibility >= 0.3:
                        mp.solutions.drawing_utils.draw_landmarks(
                            object_frame,
                            results_pose.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                        )

            # 객체 프레임을 원본 프레임에 다시 그리기
            image[c1[1] : c2[1], c1[0] : c2[0]] = object_frame

            image = cv2.rectangle(image, c1, c2, (0, 255, 0), 2)
            image = cv2.putText(
                image,
                label,
                (c1[0], c1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

# 이미지를 출력
st.image(image, caption="YOLOv5 객체 검출 및 관절 지점", use_column_width=True)
