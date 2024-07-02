import cv2
import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
import datetime
import time
import pygame
import torch
import pickle

st.set_page_config(
    page_title="실시간 3대 운동 AI 자세 교정 서비스",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# YOLOv5 모델 불러오기
model_weights_path = "./models/best_big_bounding.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_weights_path)
model.to("mps")
model.eval()

# 이전 알림 시간 기록
previous_alert_time = 0


def most_frequent(data):
    return max(data, key=data.count)


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

pygame.mixer.init()

# Sidebar에 메뉴 추가
menu_selection = st.selectbox("운동 선택", ("벤치프레스", "스쿼트", "데드리프트"))
counter_display = st.sidebar.empty()
counter_display.header(f"현재 카운터: {counter}회")

# Load different models based on the selected exercise
counter = 0
current_stage = ""
posture_status = [None]

model_weights_path = "./models/benchpress/benchpress.pkl"
with open(model_weights_path, "rb") as f:
    model_e = pickle.load(f)

if menu_selection == "벤치프레스":
    model_weights_path = "./models/benchpress/benchpress.pkl"
    with open(model_weights_path, "rb") as f:
        model_e = pickle.load(f)
elif menu_selection == "스쿼트":
    model_weights_path = "./models/squat/squat.pkl"
    with open(model_weights_path, "rb") as f:
        model_e = pickle.load(f)
elif menu_selection == "데드리프트":
    model_weights_path = "./models/deadlift/deadlift.pkl"
    with open(model_weights_path, "rb") as f:
        model_e = pickle.load(f)

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Mediapipe Pose 모델 초기화: 최소 감지 신뢰도=0.5, 최소 추적 신뢰도=0.7, 모델 복잡도=2를 준다.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.7, model_complexity=2
)

# 신뢰도 임계값 슬라이더
confidence_threshold = st.sidebar.slider("관절점 추적 신뢰도 임계값", 0.0, 1.0, 0.7)

# 각도 표시를 위한 빈 영역 초기화
neck_angle_display = st.sidebar.empty()
left_shoulder_angle_display = st.sidebar.empty()
right_shoulder_angle_display = st.sidebar.empty()
left_elbow_angle_display = st.sidebar.empty()
right_elbow_angle_display = st.sidebar.empty()
left_hip_angle_display = st.sidebar.empty()
right_hip_angle_display = st.sidebar.empty()
left_knee_angle_display = st.sidebar.empty()
right_knee_angle_display = st.sidebar.empty()
left_ankle_angle_display = st.sidebar.empty()
right_ankle_angle_display = st.sidebar.empty()

while True:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)  # 프레임 좌우반전

    # YOLOv5를 사용하여 사물 검출
    results_yolo = detect_objects(frame)

    # YOLOv5 결과를 화면에 표시
    try:
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

                    if results_pose.pose_landmarks is not None:
                        landmarks = results_pose.pose_landmarks.landmark
                        nose = [
                            landmarks[mp_pose.PoseLandmark.NOSE].x,
                            landmarks[mp_pose.PoseLandmark.NOSE].y,
                        ]  # 코
                        left_shoulder = [
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                        ]  # 좌측 어깨
                        left_elbow = [
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                        ]  # 좌측 팔꿈치
                        left_wrist = [
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y,
                        ]  # 좌측 손목
                        left_hip = [
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                        ]  # 좌측 힙
                        left_knee = [
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
                        ]  # 좌측 무릎
                        left_ankle = [
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                        ]  # 좌측 발목
                        left_heel = [
                            landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y,
                        ]  # 좌측 힐
                        right_shoulder = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                        ]  # 우측 어깨
                        right_elbow = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                        ]  # 우측 팔꿈치
                        right_wrist = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                        ]  # 우측 손목
                        right_hip = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
                        ]  # 우측 힙
                        right_knee = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                        ]  # 우측 무릎
                        right_ankle = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                        ]  # 우측 발목
                        right_heel = [
                            landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y,
                        ]  # 우측 힐

                        # 각도 계산
                        neck_angle = (
                            calculateAngle(left_shoulder, nose, left_hip)
                            + calculateAngle(right_shoulder, nose, right_hip) / 2
                        )
                        left_elbow_angle = calculateAngle(
                            left_shoulder, left_elbow, left_wrist
                        )
                        right_elbow_angle = calculateAngle(
                            right_shoulder, right_elbow, right_wrist
                        )
                        left_shoulder_angle = calculateAngle(
                            left_elbow, left_shoulder, left_hip
                        )
                        right_shoulder_angle = calculateAngle(
                            right_elbow, right_shoulder, right_hip
                        )
                        left_hip_angle = calculateAngle(
                            left_shoulder, left_hip, left_knee
                        )
                        right_hip_angle = calculateAngle(
                            right_shoulder, right_hip, right_knee
                        )
                        left_knee_angle = calculateAngle(
                            left_hip, left_knee, left_ankle
                        )
                        right_knee_angle = calculateAngle(
                            right_hip, right_knee, right_ankle
                        )
                        left_ankle_angle = calculateAngle(
                            left_knee, left_ankle, left_heel
                        )
                        right_ankle_angle = calculateAngle(
                            right_knee, right_ankle, right_heel
                        )

                        # 각도 표시 업데이트
                        neck_angle_display.text(f"Neck Angle: {neck_angle:.2f}°")
                        left_shoulder_angle_display.text(
                            f"Left Shoulder Angle: {left_shoulder_angle:.2f}°"
                        )
                        right_shoulder_angle_display.text(
                            f"Right Shoulder Angle: {right_shoulder_angle:.2f}°"
                        )
                        left_elbow_angle_display.text(
                            f"Left Elbow Angle: {left_elbow_angle:.2f}°"
                        )
                        right_elbow_angle_display.text(
                            f"Right Elbow Angle: {right_elbow_angle:.2f}°"
                        )
                        left_hip_angle_display.text(
                            f"Left Hip Angle: {left_hip_angle:.2f}°"
                        )
                        right_hip_angle_display.text(
                            f"Right Hip Angle: {right_hip_angle:.2f}°"
                        )
                        left_knee_angle_display.text(
                            f"Left Knee Angle: {left_knee_angle:.2f}°"
                        )
                        right_knee_angle_display.text(
                            f"Right Knee Angle: {right_knee_angle:.2f}°"
                        )
                        left_ankle_angle_display.text(
                            f"Left Ankle Angle: {left_ankle_angle:.2f}°"
                        )
                        right_ankle_angle_display.text(
                            f"Right Ankle Angle: {right_ankle_angle:.2f}°"
                        )

                # 횟수 세기 알고리즘 구현
                try:
                    row = [
                        coord
                        for res in results_pose.pose_landmarks.landmark
                        for coord in [res.x, res.y, res.z, res.visibility]
                    ]
                    X = pd.DataFrame([row])
                    exercise_class = model_e.predict(X)[0]
                    exercise_class_prob = model_e.predict_proba(X)[0]
                    print(exercise_class, exercise_class_prob)
                    if "down" in exercise_class:
                        current_stage = "down"
                        posture_status.append(exercise_class)
                        print(f"posture of exercise performer: {posture_status}")
                    elif current_stage == "down" and "up" in exercise_class:
                        # and exercise_class_prob[exercise_class_prob.argmax()] >= 0.3
                        current_stage = "up"
                        counter += 1
                        posture_status.append(exercise_class)
                        print(f"posture of exercise performer: {posture_status}")
                        counter_display.header(f"Current count: {counter}times")
                        if "correct" not in most_frequent(posture_status):
                            current_time = time.time()
                            if current_time - previous_alert_time >= 3:
                                now = datetime.datetime.now()
                                if "excessive_arch" in most_frequent(posture_status):
                                    options = [
                                        (
                                            "Avoid arching your lower back too much; try to keep it natural.",
                                            "./resources/sounds/excessive_arch_1.mp3",
                                        ),
                                        (
                                            "Lift your pelvis a bit and tighten your abs to keep your back flat.",
                                            "./resources/sounds/excessive_arch_2.mp3",
                                        ),
                                    ]
                                    selected_option = random.choice(options)
                                    selected_message = selected_option[0]
                                    selected_music = selected_option[1]
                                    st.error(selected_message)
                                    pygame.mixer.music.load(selected_music)
                                    pygame.mixer.music.play()
                                    posture_status = []
                                    previous_alert_time = current_time
                                elif "arms_spread" in most_frequent(posture_status):
                                    options = [
                                        (
                                            "Your grip is too wide. Hold the bar a bit narrower. ",
                                            "./resources/sounds/arms_spread_1.mp3",
                                        ),
                                        (
                                            "When gripping the bar, hold it slightly wider than shoulder width.",
                                            "./resources/sounds/arms_spread_2.mp3",
                                        ),
                                    ]
                                    selected_option = random.choice(options)
                                    selected_message = selected_option[0]
                                    selected_music = selected_option[1]
                                    st.error(selected_message)
                                    pygame.mixer.music.load(selected_music)
                                    pygame.mixer.music.play()
                                    posture_status = []
                                    previous_alert_time = current_time
                                elif "spine_neutral" in most_frequent(posture_status):
                                    options = [
                                        (
                                            "Avoid excessive curvature of the spine.",
                                            "./resources/sounds/spine_neutral_feedback_1.mp3",
                                        ),
                                        (
                                            "Lift your chest and push your shoulders back.",
                                            "./resources/sounds/spine_neutral_feedback_2.mp3",
                                        ),
                                    ]
                                    selected_option = random.choice(options)
                                    selected_message = selected_option[0]
                                    selected_music = selected_option[1]
                                    st.error(selected_message)
                                    pygame.mixer.music.load(selected_music)
                                    pygame.mixer.music.play()
                                    posture_status = []
                                    previous_alert_time = current_time
                                elif "caved_in_knees" in most_frequent(posture_status):
                                    options = [
                                        (
                                            "Be cautious not to let your knees cave in during the squat.",
                                            "./resources/sounds/caved_in_knees_feedback_1.mp3",
                                        ),
                                        (
                                            "Push your hips back to keep your knees and toes in a straight line.",
                                            "./resources/sounds/caved_in_knees_feedback_2.mp3",
                                        ),
                                    ]
                                    selected_option = random.choice(options)
                                    selected_message = selected_option[0]
                                    selected_music = selected_option[1]
                                    st.error(selected_message)
                                    pygame.mixer.music.load(selected_music)
                                    pygame.mixer.music.play()
                                    posture_status = []
                                    previous_alert_time = current_time
                                elif "feet_spread" in most_frequent(posture_status):
                                    st.error(
                                        "Narrow your stance to about shoulder width."
                                    )
                                    pygame.mixer.music.load(
                                        "./resources/sounds/feet_spread.mp3"
                                    )
                                    pygame.mixer.music.play()
                                    posture_status = []
                                    previous_alert_time = current_time
                                elif "arms_narrow" in most_frequent(posture_status):
                                    st.error(
                                        "Your grip is too wide. Hold the bar a bit narrower."
                                    )
                                    pygame.mixer.music.load(
                                        "./resources/sounds/arms_narrow.mp3"
                                    )
                                    pygame.mixer.music.play()
                                    posture_status = []
                                    previous_alert_time = current_time
                        elif "correct" in most_frequent(posture_status):
                            pygame.mixer.music.load("./resources/sounds/correct.mp3")
                            pygame.mixer.music.play()
                            st.info(
                                "You are performing the exercise with the correct posture."
                            )
                            posture_status = []
                except Exception as e:
                    pass

                # 랜드마크 그리기
                for landmark in mp_pose.PoseLandmark:
                    if landmarks[landmark.value].visibility >= confidence_threshold:
                        mp.solutions.drawing_utils.draw_landmarks(
                            object_frame,
                            results_pose.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                        )

            # 객체 프레임을 원본 프레임에 다시 그리기
            frame = object_frame

        # 원본 프레임을 출력
        FRAME_WINDOW.image(frame)
    except Exception as e:
        pass
