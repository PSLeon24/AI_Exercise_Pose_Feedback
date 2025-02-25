# Real-Time AI Posture Correction for Powerlifting Exercises Using YOLOv5 and MediaPipe
![rea](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/7a8d6fc1-2d21-45a9-b92f-fdd8cadc43b9)

- Paper: <a href="https://ieeexplore.ieee.org/abstract/document/10798440">Paper Download</a>
- PPT: <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/final_presentation.pdf">PPT Download</a>
- Demo Video: https://youtu.be/u4f_sdjk1Ig

### Description
A Study on the big three exercises AI posture correction service Using YOLOv5 and MediaPipe<br>
study duration: 2023.09.01 ~ 2023.11.20 <br>
|<b>Yeong-Min Ko</b>|
|:--:|
|<img height="180" src="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/6b04d3f5-e87a-4a2b-a2a9-e406e575b6fd">|

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
    - Deadlift
      - <a href="https://universe.roboflow.com/isbg/sdt/dataset/5">SDT Image Dataset</a>
    - More(bending, lying, sitting, standing)
      - <a href="https://www.kaggle.com/datasets/deepshah16/silhouettes-of-human-posture">Silhouettes of human posture</a>
  - Inference
    |Test 1|Test 2|
    |:---:|:---:|
    |![스크린샷 2023-09-17 154841](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/87a0b701-2df5-400b-9001-d4f526bf8211)|![스크린샷 2023-09-17 154818](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/750a4a99-cfd6-44cc-ae95-a4db42e7b67a)|
    |<b>Applied 1</b>|<b>Applied 2</b>|
    |<img width="561" alt="데드용2" src="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/e3d2e1f9-8fda-422d-9096-64d77b337b00">|<img width="564" alt="스쿼트용2" src="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/5e17f5a1-dc61-4152-82ea-24d86c563ee9">|
- Exercise Posture Correction
  - Shooting Stand Position
    |Bench Press|Squat and Deadlift|
    |:--:|:--:|
    |![KakaoTalk_Photo_2023-10-04-13-37-53 001](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/59701404-4648-497d-9893-7e617e1dd928)|![KakaoTalk_Photo_2023-10-04-13-37-54 003](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/fc403019-f221-4113-a1cf-44a5de9a042f)|
  - Example of Shooting stand position for Bench Press
    |Picture(Left, Center, Right)|
    |:--:|
    |![KakaoTalk_20231004_132437850](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/d2aec3e9-59ed-4eba-b4e5-8649cbe18260)|
  - Bench Press: <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/tree/main/labeling/benchpress">read more</a>
    - ![스크린샷 2023-10-08 192017](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/319898a4-56ba-4c6c-9b89-63b4cf885148)
  - Squat: <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/tree/main/labeling/squat">read more</a>
    - ![스크린샷 2023-10-08 203046](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/15e25bf1-5a02-4b62-ba07-65f7a1ac8bfe)
  - Deadlift: <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/tree/main/labeling/deadlift">read more</a>
    - ![스크린샷 2023-10-08 203430](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/ef3795db-e999-477c-824b-58ab4845ab55)

## Train & Evaluate
### YOLOv5
  - Detect only a Person exercising something
    - Hyperparameters to train
      - epochs 200(but early stopping: 167)
      - batch 16
      - weights yolov5s.pt
      - etc are set by 'default'
  - Performance Evaluation
    |Precision|Recall|mAP_0.5|mAP_0.5:0.95|
    |:--:|:--:|:--:|:--:|
    |0.987|0.990|0.99|0.686|
    
### Exercise Classfication
  - Bench Press (Algorithm: Random Forest)
    |Accuracy|Precision|Recall|F1-Score|
    |:--:|:--:|:--:|:--:|
    |0.961|0.963|0.961|0.961|
  - Squat (Algorithm: Random Forest)
    |Accuracy|Precision|Recall|F1-Score|
    |:--:|:--:|:--:|:--:|
    |0.989|0.989|0.989|0.989|
  - Deadlift (Algorithm: Random Forest)
    |Accuracy|Precision|Recall|F1-Score|
    |:--:|:--:|:--:|:--:|
    |0.947|0.949|0.947|0.948|

## Feedback
  |Bench Press|Squat|Deadlift|
  |:--:|:--:|:--:|
  |<b>허리가 과도한 아치 자세</b><br>허리를 너무 아치 모양으로 만들지 말고 가슴을 피려고 노력하세요.|<b>척추가 중립이 아닌 자세</b><br>척추가 과도하게 굽지 않도록 노력하세요|<b>척추 중립이 아닌 자세</b><br>척추가 과도하게 굽지 않도록 노력하세요|
  |<b>허리가 과도한 아치 자세</b><br>골반을 조금 더 들어올리고 복부를 긴장시켜 허리를 평평하게 유지하세요.|<b>척추가 중립이 아닌 자세</b><br>가슴을 들어올리고 어깨를 뒤로 넣으세요.|<b>척추가 중립이 아닌 자세</b><br>가슴을 들어올리고 어깨를 뒤로 넣으세요.|
  |<b>바를 너무 넓게 잡은 자세</b><br>바를 너무 넓게 잡았습니다. 조금 더 좁게 잡으세요.|<b>무릎이 움푹 들어간 자세</b><br>무릎이 움푹 들어가지 않도록 주의하세요.|<b>바를 너무 넓게 잡은 자세</b><br>바를 너무 넓게 잡았습니다. 조금 더 좁게 잡으세요.|
  |<b>바를 너무 넓게 잡은 자세</b><br>바를 잡을 때 어깨 너비보다 약간만 넓게 잡는 것이 좋습니다.|<b>무릎이 움푹 들어간 자세</b><br>엉덩이를 뒤로 빼서 무릎과 발끝을 일직선으로 유지하세요.|<b>바를 너무 넓게 잡은 자세</b><br>바를 잡을 때 어깨 너비보다 약간만 넓게 잡는 것이 좋습니다.|
  ||<b>발을 너무 넓게 벌린 자세</b><br>발을 어깨 너비 정도로만 벌리도록 좁히세요.|<b>바를 너무 좁게 잡은 자세</b><br>바를 어깨 너비보다 조금 넓게 잡는 것이 좋습니다.|

## How to Use
- Open your terminal in mac, linux or your command prompt in Windows. Then, type "<b>Streamlit run Streamlit.py</b>".<br>
  <img width="387" alt="스크린샷 2023-09-17 오후 4 40 44" src="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/23a85105-0836-4632-86a9-a0f87017852d">
  |This Service|
  |:---:|
  |<img width="632" alt="스크린샷 2023-12-03 오후 9 01 41" src="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/e891d8fd-3af1-4d9a-a51d-4425aa9df452">|
  
## Major project records
- 2023/09/10: 2023/09/10: Successfully concluded a project utilizing YOLOv5 to detect a singular individual.
- 2023/09/11: Integration with Mediapipe yielded lower accuracy than anticipated. Consequently, we decided to enhance labeling by introducing additional spatial dimensions around individuals.
- 2023/09/16: Significantly refined bounding boxes for model training, resulting in a triumphant pose estimation with remarkable accuracy when employing YOLOv5 and Mediapipe in tandem. Implemented a Streamlit file for holistic pose estimation after detecting the nearest person using YOLOv5. And <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/Streamlit.py">the streamlit file</a> was impleted to estimate holistic pose after detecting only person closest to the camera using yolov5.
- 2023/09/30 ~ 2023/10/02: Gathered datasets for training an exercise posture classification model.
- 2023/10/03 ~ 2023/10/08: Commenced with class labeling of the dataset, followed by model training and conclusive evaluations.
- 2023/10/18: Established a connection between the bench press model and the server, implementing an algorithm to count bench press repetitions. Additionally, in the process of linking two additional models: deadlift and squat.
- 2023/10/24: Successfully integrated all models and the server, culminating in the completion of the paper.
- 2023/11/05: Implemented feedback mechanisms for each specific posture.
- 2023/11/20: Submitted the finalized paper along with experimental results.

## Project Progress
- Week 1: Requirement Analysis
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_1.pdf">Read More</a>
- Week 2: Prototype Development & Mini Test
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_2.pptx">Read More</a>
- Week 3: Retrain the model detecting only person and Estimate holistic pose after detecting only person closest to the camera using yolov5
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_3.pdf">Read More</a>
- Week 4: Write the paper
- Week 5: Write the paper and Develop machine learning pipelines
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_5.pdf">Read More</a>
- Week 6: Presentation of project mid-progress
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/mid-term%20presentation.pdf">Read More</a>
- Week 7: Link the bench press model and the streamlit server / Implement an algorithm to count the number of bench press
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_7.pdf">Read More</a>
- Week 8: Write the paper and Link all models(bench press, squat, deadlift) and the streamlit server
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_8.pdf">Read More</a>
- Week 9: Implement feedback for each posture
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_9.pptx">Read More</a>
- Week 10: Paper Feedback
- Week 11: Paper Feedback
- Week 12: Finish the project

## Award
![우수논문상](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/73ec0496-63c6-4a10-80cc-86c20fffb3da)
