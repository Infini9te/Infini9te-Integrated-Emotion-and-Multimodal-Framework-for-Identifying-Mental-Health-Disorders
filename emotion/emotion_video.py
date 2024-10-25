import cv2
import os
import mediapipe as mp
import time
from tensorflow import keras
from keras.applications.resnet import preprocess_input
import numpy as np
import json
from datetime import datetime
from tkinter import Tk, filedialog  # 附加檔案對話框模組
os.chdir(r'C:\Users\NHRI\Desktop\情緒與專注力\情緒')
def load_model(model_json_path, model_weights_path):
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_path)
    return loaded_model

def main():
    start_time = time.time()
    
    # 使用檔案對話框選擇視訊文件
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi")])
    root.destroy()

    #video_path = r'C:\Users\z3412\Desktop\emotion\new code\0703\MMHT000187_Speech_60.mp4'
    #video_path = 'D:/Livia/python/test/MMHT000187_Speech_60FPS24.mp4'
    #video_path = 'D:/Livia/python/test/MMHT000187_Speech_60FPS40.mp4'
    
    video_capture = cv2.VideoCapture(video_path)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0]
    video_folder = os.path.dirname(video_path)
    
    # Automatically detect paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_json_path = os.path.join(script_dir, 'model_json.json')
    model_weights_path = os.path.join(script_dir, 'model_weights.h5')
    
    model = load_model(model_json_path, model_weights_path)
    
    # 初始化 MediaPipe 人臉檢測器和人臉關鍵特徵點檢測器
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    json_name="emotion.json"
    open(json_name, "w").close()
    json_name2="time.json"
    open(json_name2, "w").close()

    positive_count = 0
    negative_count = 0
    neutral_count = 0
    frame_count = 0
    seconds = 0
    Count = [0,0,0,0]
    emotion_per_second = {}
    result_dict = {}
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 2 == 0:# 1/2 frame 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 將圖像傳遞給 MediaPipe 人臉檢測器進行檢測
            results_detection = face_detection.process(frame_rgb)

            # 如果檢測到人臉，則繪製邊界框和關鍵特徵點
            if results_detection.detections:
                for detection in results_detection.detections:
                    # 將圖像傳遞給 MediaPipe 人臉關鍵特徵點檢測器進行檢測
                    results_mesh = face_mesh.process(frame_rgb)

                    # 如果檢測到人臉關鍵特徵點
                    if results_mesh.multi_face_landmarks:
                        # Preprocess frame
                        processed_frame = cv2.resize(frame, (48, 48))
                        processed_frame = np.expand_dims(processed_frame, axis=0)
                        processed_frame = preprocess_input(processed_frame)

                        # Predict emotion
                        prediction = model.predict(processed_frame)
                        emotion_label = np.argmax(prediction)

                        if emotion_label == 0:
                            negative_count += 1
                        elif emotion_label == 1:
                            neutral_count += 1
                        elif emotion_label == 2:
                            positive_count += 1
            
        if frame_count % fps == 0: 
            key = f"{seconds}s"      
            if negative_count== 0 and neutral_count ==0 and positive_count ==0:
                Count[3] += 1             
                emotion_per_second[key] = "Unknown" 
            else:     
                max_count = max(positive_count, negative_count, neutral_count)
                if max_count == positive_count:
                    Count[0] += 1
                    emotion_per_second[key] = "Positive" 
                elif max_count == neutral_count:
                    Count[1] += 1
                    emotion_per_second[key] = "Neutral" 
                elif max_count == negative_count:
                    Count[2] += 1
                    emotion_per_second[key] = "Negative"
            
            with open(json_name2, "w") as file:
                json.dump(emotion_per_second, file, indent=4)
                file.write("\n")
    
            seconds += 1    
            positive_count = 0
            negative_count = 0
            neutral_count = 0  
    
    result_dict["Video_ID"] = video_id
    result_dict["Positive"] = Count[0]
    result_dict["Neutral"] = Count[1]
    result_dict["Negative"] = Count[2]
    result_dict["Unknown"] = Count[3]
    result_dict["Total"] = seconds
    result_dict["Positive Ratio"] = round((Count[0] / (seconds)) * 100, 1)
    result_dict["Neutral Ratio"] = round((Count[1] / (seconds)) * 100, 1)
    result_dict["Negative Ratio"] = round((Count[2] / (seconds)) * 100, 1)

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y/%m/%d %H:%M:%S")

    result_dict["Execution Datetime"] = formatted_datetime

    with open(json_name, "w") as file:
        json.dump(result_dict, file, indent=4)

    video_capture.release()
    
    
    #### label to video ####
    with open(json_name2, "r") as file:
        emotion_data = json.load(file)

    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_name = os.path.join(video_folder, f"1_output.mp4")
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_count += 1
           
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_detection = face_detection.process(frame_rgb) 

        if results_detection.detections:
            for detection in results_detection.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)
                        
                results_mesh = face_mesh.process(frame_rgb)

                if results_mesh.multi_face_landmarks:
                    seconds_key = f"{frame_count // fps}s" 
                    
                    if (frame_count // fps) == len(emotion_data):
                        seconds_key = f"{(frame_count // fps) - 1}s"
                            
                    emotion_text = emotion_data.get(seconds_key, "Unknown")
                
                    if emotion_text == 'Positive':
                        emotion_text = 'Positive'
                        text_color = (0, 255, 0)  # green color
                    elif emotion_text == 'Negative':
                        emotion_text = 'Negative'
                        text_color = (0, 0, 255)  # red color
                    elif emotion_text == 'Neutral':
                        emotion_text = 'Neutral'
                        text_color = (255, 0, 0)  # blue color 
                    elif emotion_text == 'Unknown':
                        emotion_text = 'Unknown'
                        text_color = (0, 0, 0)  # black color
                        
                    cv2.rectangle(frame, (x, y), (x+w, y+h), text_color, 2)
                    cv2.putText(frame, f"Emotion: {emotion_text}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2) 
                    
        out.write(frame)

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
        
    end_time = time.time()
    total_time = end_time - start_time
    print("Spend time:", total_time, "s")

    
if __name__ == "__main__":
    main()