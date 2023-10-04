# AI_Exercise_Pose_Feedback
### Description
A Study on the big three exercises AI posture correction service Using YOLOv5 and MediaPipe<br>
<b>Yeong-Min Ko</b>
### Development Environment
- OS: MAC m1 & Windows 11(NVIDIA GeForce RTX 4080 Ti)<br>
- Frameworks & Libraries: YOLOv5, MediaPipe, OpenCV, Streamlit
- Device: iPhone 12 Pro(WebCam using iVCam)

## Data
- YOLOv5: <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/tree/main/yolov5_onlyPerson">Detect only one Person</a>
  - Scraping Images from Google & Roboflow
    - Bench Press
      - Google Search Keyword: Bench Press
      - <a href="https://universe.roboflow.com/jiangsu-ocean-universit/faller">Faller Computer Vision Project</a> with lying down
    - Squat
      - <a href="https://universe.roboflow.com/nejc-graj-1na9e/squat-depth/dataset/14/download">Squat-Depth Image Dataset</a>
      - <a href="https://universe.roboflow.com/models/object-detection">HumonBody1 Computer Vision Project</a> with Standing
    - Dead Lift
      - <a href="https://universe.roboflow.com/isbg/sdt/dataset/5">SDT Image Dataset</a>
    - More(bending, lying, sitting, standing)
      - <a href="https://www.kaggle.com/datasets/deepshah16/silhouettes-of-human-posture">Silhouettes of human posture</a>
  - Inference
    |test1|test2|
    |---|---|
    |![스크린샷 2023-09-17 154841](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/87a0b701-2df5-400b-9001-d4f526bf8211)|![스크린샷 2023-09-17 154818](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/750a4a99-cfd6-44cc-ae95-a4db42e7b67a)|
- Exercise Posture Correction
  - Shooting Stand Position
    |Bench Press|Squat and Deadlift|
    |:--:|:--:|
    |![KakaoTalk_Photo_2023-10-04-13-37-53 001](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/59701404-4648-497d-9893-7e617e1dd928)|![KakaoTalk_Photo_2023-10-04-13-37-54 003](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/fc403019-f221-4113-a1cf-44a5de9a042f)|
  - Example of Shooting stand position for Bench Press
    |Picture(Left, Center, Right)|
    |:--:|
    |![KakaoTalk_20231004_132437850](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/d2aec3e9-59ed-4eba-b4e5-8649cbe18260)|
  - Bench Press
    
  - Squat
    
  - Dead Lift

## How to Use
- Open your terminal in mac, linux or your command prompt in Windows. Then, type "Streamlit run Streamlit.py".
  <img width="387" alt="스크린샷 2023-09-17 오후 4 40 44" src="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/23a85105-0836-4632-86a9-a0f87017852d">
  |picture1|picture2|
    |---|---|
    |![스크린샷 2023-09-16 222045](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/d4c754b3-9569-4969-82cb-18bebbc6f9dd)|![스크린샷 2023-09-16 224645](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/95129ae5-9cf3-41a3-90a4-017150efeb1c)|
## Major project records
- 2023/09/10: The project detecting only one person using yolov5 was completed.
- 2023/09/11: As a result of using it in combination with mediapipe, the accuracy was lower than expected. Therefore, we plan to do labeling by adding extra space around people.
- 2023/09/16: The bounding box to train the model was significantly relabeled, and as a result, pose estimation was finally successful with high accuracy when using yolov5 and mediapipe together. And <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/Streamlit.py">the streamlit file</a> was impleted to estimate holistic pose after detecting only person closest to the camera using yolov5.
- 2023/09/30 ~ 2023/10/02: I have collected dataset to train exercise posture classification model.

## Project Progress
- Week1: Requirement Analysis
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_1.pdf">Read More</a>
- Week2: Prototype Development & Mini Test
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_2.pptx">Read More</a>
- Week3: Retrain the model detecting only person and Estimate holistic pose after detecting only person closest to the camera using yolov5
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_3.pdf">Read More</a>
- Week4: Write the paper
- Week5: Write the paper and Develop machine learning pipelines
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_5.pdf">Read More</a>
