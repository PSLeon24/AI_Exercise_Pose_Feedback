# AI_Exercise_Pose_Feedback
### Description
A Study on the big three exercises AI posture correction service Using YOLOv5 and MediaPipe<br>
<b>Yeong-Min Ko</b>
### development environment
- OS: MAC m1 & Windows 11<br>
- Frameworks & Libraries: YOLOv5, MediaPipe, OpenCV, Streamlit

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
- Exercise Posture Correction

## How to Use
- To be added

## Major project records
- 2023/09/10: The project detecting only one person using yolov5 was completed.
- 2023/09/11: As a result of using it in combination with mediapipe, the accuracy was lower than expected. Therefore, we plan to do labeling by adding extra space around people.

## Project Progress
- Week1: Requirement Analysis
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_1.pdf">Read More</a>
- Week2: Prototype Development & Mini Test
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_2.pptx">Read More</a>
