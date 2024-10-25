import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import math
import json


def processed_mediapipe(file_path):
    start_time = time.time()
    CEF_COUNTER =0
    count = [0,0]
    frame_count = 0
    Record = {}
    seconds = 0
    not_gaze_count = 0
    gaze_count = 0
    result_dict = {}

    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
    RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

    # 初始化 MediaPipe Face Mesh
    map_face_mesh = mp.solutions.face_mesh
    facemesh = map_face_mesh.FaceMesh(max_num_faces=2)

    #camera = cv.VideoCapture("childeye.mp4")
    # camera = cv.VideoCapture("MMHT000380_Speech_72.mp4")
    camera = cv.VideoCapture(file_path)

    output_filename = 'mediapipe_eye2.mp4'
    frame_width = int(camera.get(3)) 
    frame_height = int(camera.get(4)) 
    fps = int(camera.get(5))
    fps2 = int(camera.get(cv.CAP_PROP_FPS))  
    out = cv.VideoWriter(output_filename, cv.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # 定義函式，用於檢測面部關鍵點
    def landmarksDetection(img, results, draw=False):
        img_height, img_width= img.shape[:2]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
        return mesh_coord

    # 定義函式，計算兩點間的歐式距離
    def euclaideanDistance(point1, point2):
        x, y = point1
        x1, y1 = point2
        distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
        return distance

    # 定義函式，計算眨眼比率
    def blinkRatio(img, landmarks, right_indices, left_indices):
        rh_right = landmarks[right_indices[0]]
        rh_left = landmarks[right_indices[8]]
        rv_top = landmarks[right_indices[12]]
        rv_bottom = landmarks[right_indices[4]]

        lh_right = landmarks[left_indices[0]]
        lh_left = landmarks[left_indices[8]]
        lv_top = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]
        
        rhDistance = euclaideanDistance(rh_right, rh_left)
        rvDistance = euclaideanDistance(rv_top, rv_bottom)
        lvDistance = euclaideanDistance(lv_top, lv_bottom)
        lhDistance = euclaideanDistance(lh_right, lh_left)

        reRatio = rhDistance/rvDistance
        leRatio = lhDistance/lvDistance
        ratio = (reRatio+leRatio)/2
        return ratio 


    while True:
        ret, frame = camera.read() 
        if not ret: 
            break
        
        frame_count  += 1
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        
        # 進行面部關鍵點檢測
        results  = facemesh.process(rgb_frame)     
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

            if ratio >3.2:
                CEF_COUNTER +=1
                if CEF_COUNTER >= fps:
                    cv.putText(frame, f'Blinking: Not Gaze', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    count[0] += 1
                    
            if ratio < 3.5:
                CEF_COUNTER = 0
                cv.putText(frame, f'Blinking: Gaze', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
                count[1] += 1
            else:
                pass
            
            if(CEF_COUNTER<fps and CEF_COUNTER >0):
                cv.putText(frame, f'Blinking: Gaze', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
                count[1] += 1
        
            BlinkingText = f"Blinking: {CEF_COUNTER}"
            cv.putText(frame, BlinkingText, (50, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            if frame_count % fps == 0:
                key = f"{seconds}s"
                max_count = max(count[0], count[1])
                if max_count == count[0]:
                    Record[key] = "NOT GAZE"
                    not_gaze_count += 1
                else:
                    Record[key] = "GAZE"
                    gaze_count += 1
                count = [0,0]
                seconds += 1
                
                output_file = 'mediapipe.json'
                
                with open(output_file, 'w') as file:
                    json.dump(Record, file, indent=4)
                result_dict["NOT GAZE"] = not_gaze_count
                result_dict["GAZE"] = gaze_count
                result_dict["Total"] = seconds
                if seconds == 0:
                    result_dict["EYES CLOSED"] = 0
                    result_dict["EYES OPENED"] = 0
                else:
                    result_dict["EYES NOT GAZE"] = round((not_gaze_count / (seconds)) * 100, 1)
                    result_dict["EYES GAZE"] = round((gaze_count / (seconds)) * 100, 1)
                output_file2 = 'mediapipe2.json'
                with open(output_file2, 'w') as file:
                    json.dump(result_dict, file, indent=4)
        
        out.write(frame)
        

    end_time = time.time()
    execution_time = round((end_time - start_time),1)
    print("程式執行時間：", execution_time, "秒")  
                
    cv.destroyAllWindows()
    camera.release()
    return result_dict,Record