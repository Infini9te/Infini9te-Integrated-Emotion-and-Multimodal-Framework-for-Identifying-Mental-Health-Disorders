import cv2
import mediapipe as mp
import numpy as np
import json
from gaze_tracking import GazeTracking
import warnings 
import os

os.chdir(r"C:\Users\NHRI\Desktop\情緒與專注力\專注力\注視")

warnings.filterwarnings("ignore",category=UserWarning)
# 設置這些值以顯示/隱藏估算的特定向量
draw_gaze = False # 是否繪製注視向量
draw_headpose = False # 是否繪製頭部姿勢向量

# 注視分數乘數（較高的乘數=注視對頭部姿勢估算的影響更大）
x_score_multiplier = 10
y_score_multiplier = 10

# 幀間平均分數的閾值
threshold = .3

# 初始化 Mediapipe 的 FaceMesh 模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,refine_landmarks=True,max_num_faces=2,min_detection_confidence=0.5)

gaze = GazeTracking()
cap = cv2.VideoCapture('childeye.mp4')
# cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_filename = 'gaze.mp4'
out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# 定義 3D 面部特徵點
face_3d = np.array([
    [0.0, 0.0, 0.0],            # 鼻子尖端
    [0.0, -330.0, -65.0],       # 下巴
    [-225.0, 170.0, -135.0],    # 左眼左角
    [225.0, 170.0, -135.0],     # 右眼右角
    [-150.0, -150.0, -125.0],   # 左嘴角
    [150.0, -150.0, -125.0]     # 右嘴角
    ], dtype=np.float64)

# 將左眼角重定位為原點
leye_3d = np.array(face_3d)
leye_3d[:,0] += 225
leye_3d[:,1] -= 175
leye_3d[:,2] += 135

# 將右眼角重定位為原點
reye_3d = np.array(face_3d)
reye_3d[:,0] -= 225
reye_3d[:,1] -= 175
reye_3d[:,2] += 135

# 上一幀的注視分數
last_lx, last_rx = 0, 0
last_ly, last_ry = 0, 0

blinking_count=0
result_data = {}
Record = {}
Record_valid = {}
correct = [0,0]
count = [0,0]

frame_count = 0
gaze_gaze_count  = 0
gaze_notgaze_count = 0
valid_gaze_count = 0
valid_notgaze_count = 0



with open("Gaze_Data_for_Comparision.json","w") as json_file:
    time=0
    seconds = 0
    while True:
        success, img = cap.read()
        if not success:
            break
        frame_count += 1
    # added by kgpt
        time+=1
        data = {}
        data["Time of Capturing in 1/10th of seconds"]=time
    # added by kgpt
    
        img=cv2.flip(img,1)
        img = cv2.resize(img,(1200,800))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img.flags.writeable = False

        results = face_mesh.process(img)
        img.flags.writeable = True

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        (img_h, img_w, img_c) = img.shape
        face_2d = []

        # 如果沒有偵測到面部，則繼續下一個迭代
        if not results.multi_face_landmarks:
            out.write(img)
            frame_without_face = img.copy()
            text_3 = "Unknown"
            count[1] += 1
            correct[1] += 1
            cv2.putText(frame_without_face, "Our Code Gaze :" + text_3, (5, 200), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
            out.write(frame_without_face)
            
            data = {
                "Time of Capturing in 1/10th of seconds": time,
                "Our Code Output": text_3,
                "Verification Method Output":text_3
            }
            json.dump(data, json_file, indent=1)
            
            if frame_count % fps == 0:
                key = f"{seconds}s Our Code :"
                key2 = f"{seconds}s Verification :"
                max_count = max(count[0], count[1])
                max_count2 = max(correct[0],correct[1])
                
                    
                if max_count == count[0]:
                    valid_gaze_count += 1
                    Record_valid[key2] = "GAZE"
                elif max_count == count[1]:
                    valid_notgaze_count += 1
                    Record_valid[key2] = "NOT GAZE"
                
                    
                if max_count2 == correct[0]:
                    gaze_gaze_count += 1
                    Record[key] = "GAZE"
                elif max_count2 == correct[1]:
                    gaze_notgaze_count += 1
                    Record[key] = "NOT GAZE" 
                
                seconds += 1
                correct = [0,0]            
                count = [0,0]
                
                
            continue

        for face_landmarks in results.multi_face_landmarks:
            face_2d = []            
            for idx, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * img_w), int(lm.y * img_h)# 將特徵點 x 和 y 轉換為像素座標
                face_2d.append((x, y))# 將 2D 座標添加到數組中
            
            # 獲取用於頭部姿勢估算的相關特徵點
            face_2d_head = np.array([
                face_2d[1],      # 鼻子
                face_2d[199],    # 下巴
                face_2d[33],     # 左眼左角
                face_2d[263],    # 右眼右角
                face_2d[61],     # 左嘴角
                face_2d[291]     # 右嘴角
            ], dtype=np.float64)

            face_2d = np.asarray(face_2d)

            # 計算 left x gaze score
            if (face_2d[243,0] - face_2d[130,0]) != 0:
                lx_score = (face_2d[468,0] - face_2d[130,0]) / (face_2d[243,0] - face_2d[130,0])
                if abs(lx_score - last_lx) < threshold:
                    lx_score = (lx_score + last_lx) / 2
                last_lx = lx_score

            # 計算 left y gaze score
            if (face_2d[23,1] - face_2d[27,1]) != 0:
                ly_score = (face_2d[468,1] - face_2d[27,1]) / (face_2d[23,1] - face_2d[27,1])
                if abs(ly_score - last_ly) < threshold:
                    ly_score = (ly_score + last_ly) / 2
                last_ly = ly_score

            # 計算 right x gaze score
            if (face_2d[359,0] - face_2d[463,0]) != 0:
                rx_score = (face_2d[473,0] - face_2d[463,0]) / (face_2d[359,0] - face_2d[463,0])
                if abs(rx_score - last_rx) < threshold:
                    rx_score = (rx_score + last_rx) / 2
                last_rx = rx_score

            # 計算 right y gaze score
            if (face_2d[253,1] - face_2d[257,1]) != 0:
                ry_score = (face_2d[473,1] - face_2d[257,1]) / (face_2d[253,1] - face_2d[257,1])
                if abs(ry_score - last_ry) < threshold:
                    ry_score = (ry_score + last_ry) / 2
                last_ry = ry_score

            # 獲取相機矩陣
            focal_length = 1 * img_w
            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # Distortion coefficients 
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            _, l_rvec, l_tvec = cv2.solvePnP(leye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            _, r_rvec, r_tvec = cv2.solvePnP(reye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            # 從旋轉向量獲取旋轉矩陣
            l_rmat, _ = cv2.Rodrigues(l_rvec)
            r_rmat, _ = cv2.Rodrigues(r_rvec)

            # [0] changes pitch [1] changes roll [2] changes yaw
            # +1 更改 ~45 度（俯仰向下，滾動向左傾斜（逆時針），偏航向左旋轉（逆時針））
            # 用注視分數調整頭部姿勢向量
            l_gaze_rvec = np.array(l_rvec)
            l_gaze_rvec[2][0] -= (lx_score-.5) * x_score_multiplier
            l_gaze_rvec[0][0] += (ly_score-.5) * y_score_multiplier

            r_gaze_rvec = np.array(r_rvec)
            r_gaze_rvec[2][0] -= (rx_score-.5) * x_score_multiplier
            r_gaze_rvec[0][0] += (ry_score-.5) * y_score_multiplier


            # 獲取左眼角（整數）
            l_corner = face_2d_head[2].astype(np.int32)
            l_corner_x = face_2d_head[2][0]  # 眼角的 x 座標
            l_corner_y = face_2d_head[2][1]  # 眼角的 y 座標

            # 將旋轉軸投影到左眼的平面上
            axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
            l_axis, _ = cv2.projectPoints(axis, l_rvec, l_tvec, cam_matrix, dist_coeffs)
            l_gaze_axis, _ = cv2.projectPoints(axis, l_gaze_rvec, l_tvec, cam_matrix, dist_coeffs)
            l_axis_x, l_axis_y = tuple(np.ravel(l_axis[2]).astype(np.int32))
            l_gaze_axis_x, l_gaze_axis_y = tuple(np.ravel(l_gaze_axis[2]).astype(np.int32))

            # 繪製左眼旋轉軸
            if draw_headpose:
                cv2.line(img, l_corner, tuple(np.ravel(l_axis[2]).astype(np.int32)), (0,200,200), 3)
        
            if draw_gaze:
                cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)

    
            # 獲取右眼角（整數）
            r_corner = face_2d_head[3].astype(np.int32)
            r_corner_x = face_2d_head[3][0]  # 眼角的 x 座標
            r_corner_y = face_2d_head[3][1]  # 眼角的 y 座標
            r_axis, _ = cv2.projectPoints(axis, r_rvec, r_tvec, cam_matrix, dist_coeffs)
            r_gaze_axis, _ = cv2.projectPoints(axis, r_gaze_rvec, r_tvec, cam_matrix, dist_coeffs)
            r_axis_x, r_axis_y = tuple(np.ravel(r_axis[2]).astype(np.int32))
            r_gaze_axis_x, r_gaze_axis_y = tuple(np.ravel(r_gaze_axis[2]).astype(np.int32))

            # 繪製右眼旋轉軸
            if draw_headpose:
                cv2.line(img, r_corner, tuple(np.ravel(r_axis[2]).astype(np.int32)), (0,200,200), 3)

            if draw_gaze:
                cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)
                        
                        
            # 獲取左眼注視向量和右眼注視向量的中心線
            gaze_center_x = int((l_gaze_axis_x + r_gaze_axis_x) / 2)
            gaze_center_y = int((l_gaze_axis_y + r_gaze_axis_y) / 2)
            center_x = int((l_corner_x + r_corner_x)/2)
            center_y = int((l_corner_y + r_corner_y)/2)
            
            # 將整數轉換為字串
            center_x_str = str(center_x)
            center_y_str = str(center_y)
            gaze_center_x_str = str(gaze_center_x)
            gaze_center_y_str = str(gaze_center_y)
            
            diff_X = center_x - gaze_center_x
            diff_Y = center_y - gaze_center_y
            diff_X_str = str(diff_X)
            diff_Y_str = str(diff_Y)
            
            if 0 > diff_Y > -100:
                text_2="GAZE"
                correct[0] += 1
                #cv2.line(img, (center_x, center_y), (gaze_center_x, gaze_center_y), (0, 255, 0), 3) 
            else:
                text_2="NOT GAZE"
                correct[1] += 1

# added by kgpt
            cv2.putText(img,"Our Code Gaze : "+text_2, (5,200), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)       
            data["Our Code Output"]=text_2
# added by kgpt
    
            # 繪製左右眼注視向量的中心線
            # cv2.line(img, (center_x, center_y), (gaze_center_x, gaze_center_y), (0, 255, 0), 3) 
            cv2.putText(img, center_x_str + ", " + center_y_str, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, gaze_center_x_str + ", " + gaze_center_y_str, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)       
            # cv2.putText(img, diff_X_str + ", " + diff_Y_str, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)       

    # added by kgpt
        # We send this frame to GazeTracking to analyze it
        gaze.refresh(img)

        text = "GAZE"
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        try:
            # if gaze.is_top():
            #     text = "NOT GAZE"
            #     count[1] += 1
            # elif gaze.is_bottom():
            #     text = "NOT GAZE"
            #     count[1] += 1
            if gaze.is_right():
                text = "NOT GAZE"
                count[1] += 1
                # cv2.putText(img, "Gaze Verification : "+ text , (5,100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,0,0), 1)
            elif gaze.is_left():
                text = "NOT GAZE"
                count[1] += 1
                # cv2.putText(img, "Gaze Verification : "+ text , (5,100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,0,0), 1)
            elif gaze.is_center():
                text = "GAZE"
                count[0] += 1
            elif gaze.is_blinking():
                text = "Blinking"
                blinking_count+=1
        except:
            pass

        # cv2.putText(img, "Gaze Verification : "+ text , (5,100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,0,0), 1)
        data["Verification Method Output"]=text
        json.dump(data,json_file,indent=1)
        if frame_count % fps == 0:
            key = f"{seconds}s Our Code :"
            key2 = f"{seconds}s Verification :"
            record = {}
            record["seconds"] = seconds
            max_count = max(count[0], count[1])
            max_count2 = max(correct[0],correct[1])
            if max_count == count[0]:
                valid_gaze_count += 1
                Record_valid[key2] = "GAZE"
            elif max_count == count[1]:
                valid_notgaze_count += 1
                Record_valid[key2] = "NOT GAZE"
            else: 
                valid_notgaze_count += 1
                Record_valid[key2] = "NOT GAZE"
                
            if max_count2 == correct[0]:
                gaze_gaze_count += 1
                Record[key] = "GAZE"
            elif max_count2 == correct[1]:
                gaze_notgaze_count += 1
                Record[key] = "NOT GAZE"
            else:
                gaze_notgaze_count += 1
                Record[key] = "NOT GAZE"
            
            seconds += 1
            correct = [0,0]            
            count = [0,0]                   
        #out.write(img)
        cv2.imshow("Gaze Positon Estimation", img)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    result_data["Original GAZE"] = gaze_gaze_count
    result_data["Original NOTGAZE"] = gaze_notgaze_count
    result_data["Verified GAZE"] = valid_gaze_count
    result_data["Verified NOTGAZE"] = valid_notgaze_count
    result_data["Seconds"] = seconds
    if seconds == 0:
        result_data["Original GAZE Ratio"] = 0
        result_data["Original NOTGAZE Ratio"] = 0
        result_data["Verified GAZE Ratio"] = 0
        result_data["Verified NOTGAZE Ratio"] = 0
    else:
        result_data["Original GAZE Ratio"] = round((gaze_gaze_count / (seconds)) * 100, 1)
        result_data["Original NOTGAZE Ratio"] = round((gaze_notgaze_count / (seconds)) * 100, 1)
        result_data["Verified GAZE Ratio"] = round((valid_gaze_count / (seconds)) * 100, 1)
        result_data["Verified NOTGAZE Ratio"] = round((valid_notgaze_count/ (seconds)) * 100, 1)
    
    output_file2 = 'gaze_data.json'
    with open(output_file2, 'w') as file:
        json.dump(result_data, file, indent=4)   
                
    output_file = 'gaze_time_record_one.json'
    with open(output_file, 'w') as file:
        json.dump(Record, file, indent=4)
        
    output_file3 = 'gaze_time_record_valid_one.json'
    with open(output_file3, 'w') as file:
        json.dump(Record_valid, file, indent=4)
        
cap.release()
cv2.destroyAllWindows()
# json.dump(f"Total Number of Eye Blinking : {blinking_count}",json_file,indent=1)
json_file.close()
print("Succesfully Done")