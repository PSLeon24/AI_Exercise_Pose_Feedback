# AI_Exercise_Pose_Feedback
![rea](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/7a8d6fc1-2d21-45a9-b92f-fdd8cadc43b9)

- Paper: <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/PAPER/YOLOv5%20%EB%B0%8F%20MediaPipe%EB%A5%BC%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%EC%8B%A4%EC%8B%9C%EA%B0%84%203%EB%8C%80%20%EC%9A%B4%EB%8F%99%20AI%20%EC%9E%90%EC%84%B8%20%EA%B5%90%EC%A0%95%20%EC%84%9C%EB%B9%84%EC%8A%A4%EC%97%90%20%EB%8C%80%ED%95%9C%20%EC%97%B0%EA%B5%AC.pdf">Paper Download</a>
- PPT: <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/final_presentation.pdf">PPT Download</a>

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
    |test1|test2|
    |:---:|:---:|
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
    |<img width="1439" alt="스크린샷 2023-11-05 오후 5 16 02" src="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/c815206c-0aae-4b4f-bf2b-4032be70e0cb">|
  
## Major project records
- 2023/09/10: The project detecting only one person using yolov5 was completed.
- 2023/09/11: As a result of using it in combination with mediapipe, the accuracy was lower than expected. Therefore, we plan to do labeling by adding extra space around people.
- 2023/09/16: The bounding box to train the model was significantly relabeled, and as a result, pose estimation was finally successful with high accuracy when using yolov5 and mediapipe together. And <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/Streamlit.py">the streamlit file</a> was impleted to estimate holistic pose after detecting only person closest to the camera using yolov5.
- 2023/09/30 ~ 2023/10/02: I have collected dataset to train exercise posture classification model.
- 2023/10/03 ~ 2023/10/08: First, I have labeled the dataset's class. And then I have trained models. Finally, I have evaluated the models.
- 2023/10/18: I linked the bench press model and the server and implemented an algorithm to count the number of doing bench presses. Additionally, I will be linking two models: deadlift and squat.
- 2023/10/24: I linked all models and the server. I have finished writing the paper.
- 2023/11/05: I implemented feedback for each posture
- 2023/11/20: I submitted the paper and experiment results.

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
