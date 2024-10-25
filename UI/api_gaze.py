import cv2
import mediapipe as mp
import numpy as np
import dlib
from math import hypot
import json
import time

def processed_gaze(file_path):
    video_inptut = file_path
    cap = cv2.VideoCapture(video_inptut)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    start_time = time.time()
    ############## PARAMETERS #######################################################

    # Set these values to show/hide certain vectors of the estimation
    draw_gaze = False
    draw_full_axis = True
    draw_headpose = False
    

    # Gaze Score multiplier (Higher multiplier = Gaze affects headpose estimation more)
    x_score_multiplier = 10
    y_score_multiplier = 10

    # Threshold of how close scores should be to average between frames
    threshold = .3

    #參數
    Record = {}
    result_data = {}
    frame_count = 0
    seconds = 0
    left_count = 0
    right_count = 0
    center_count = 0
    out_of_range_count = 0
    count = [0,0,0,0]


    #################################################################################

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5)


    face_3d = np.array([
        [0.0, 0.0, 0.0],            # Nose tip
        [0.0, -330.0, -65.0],       # Chin
        [-225.0, 170.0, -135.0],    # Left eye left corner
        [225.0, 170.0, -135.0],     # Right eye right corner
        [-150.0, -150.0, -125.0],   # Left Mouth corner
        [150.0, -150.0, -125.0]     # Right mouth corner
        ], dtype=np.float64)

    # Reposition left eye corner to be the origin
    leye_3d = np.array(face_3d)
    leye_3d[:,0] += 225
    leye_3d[:,1] -= 175
    leye_3d[:,2] += 135

    # Reposition right eye corner to be the origin
    reye_3d = np.array(face_3d)
    reye_3d[:,0] -= 225
    reye_3d[:,1] -= 175
    reye_3d[:,2] += 135

    # Gaze scores from the previous frame
    last_lx, last_rx = 0, 0
    last_ly, last_ry = 0, 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def midpoint(p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    font = cv2.FONT_HERSHEY_PLAIN

    def get_blinking_ratio(eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

        #hor_line = cv2.line(img, left_point, right_point, (0, 255, 0), 2)
        #ver_line = cv2.line(img, center_top, center_bottom, (0, 255, 0), 2)

        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

        ratio = hor_line_length / ver_line_length
        return ratio

    def get_gaze_ratio(eye_points, facial_landmarks):
        left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
        #cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

        height, width, _ = img.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        height, width = threshold_eye.shape
        
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)

        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        if left_side_white == 0:
            gaze_ratio = 1
        elif right_side_white == 0:
            gaze_ratio = 5
        else:
            gaze_ratio = left_side_white / right_side_white
        return gaze_ratio

    def distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


    #while cap.isOpened():
    while True:    
        success, img = cap.read()
        frame_count += 1
        
        if not success:
            break
        
        # Flip + convert img from BGR to RGB
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # To improve performance
        img.flags.writeable = False
        
        # Get the result
        results = face_mesh.process(img)
        img.flags.writeable = True
        
        if img is None or img.size == 0:
            raise ValueError("Failed")        
        
        # Convert the color space from RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        (img_h, img_w, img_c) = img.shape
        face_2d = []

        

        if not results.multi_face_landmarks:
            cv2.putText(img, "Not Gaze", (50, 50), font, 2, (0, 0, 255), 3)
            count[3] += 1
            continue
        
        faces = detector(gray)
        
        for face_landmarks,face in zip(results.multi_face_landmarks,faces):
            face_2d = []
            for idx, lm in enumerate(face_landmarks.landmark):
                # Convert landmark x and y to pixel coordinates
                x, y = int(lm.x * img_w), int(lm.y * img_h)

                # Add the 2D coordinates to an array
                face_2d.append((x, y))
            _ = idx
            # Get relevant landmarks for headpose estimation
            face_2d_head = np.array([
                face_2d[1],      # Nose
                face_2d[199],    # Chin
                face_2d[33],     # Left eye left corner
                face_2d[263],    # Right eye right corner
                face_2d[61],     # Left mouth corner
                face_2d[291]     # Right mouth corner
            ], dtype=np.float64)

            face_2d = np.asarray(face_2d)

            # Calculate left x gaze score
            if (face_2d[243,0] - face_2d[130,0]) != 0:
                lx_score = (face_2d[468,0] - face_2d[130,0]) / (face_2d[243,0] - face_2d[130,0])
                if abs(lx_score - last_lx) < threshold:
                    lx_score = (lx_score + last_lx) / 2
                last_lx = lx_score

            # Calculate left y gaze score
            if (face_2d[23,1] - face_2d[27,1]) != 0:
                ly_score = (face_2d[468,1] - face_2d[27,1]) / (face_2d[23,1] - face_2d[27,1])
                if abs(ly_score - last_ly) < threshold:
                    ly_score = (ly_score + last_ly) / 2
                last_ly = ly_score

            # Calculate right x gaze score
            if (face_2d[359,0] - face_2d[463,0]) != 0:
                rx_score = (face_2d[473,0] - face_2d[463,0]) / (face_2d[359,0] - face_2d[463,0])
                if abs(rx_score - last_rx) < threshold:
                    rx_score = (rx_score + last_rx) / 2
                last_rx = rx_score

            # Calculate right y gaze score
            if (face_2d[253,1] - face_2d[257,1]) != 0:
                ry_score = (face_2d[473,1] - face_2d[257,1]) / (face_2d[253,1] - face_2d[257,1])
                if abs(ry_score - last_ry) < threshold:
                    ry_score = (ry_score + last_ry) / 2
                last_ry = ry_score

            # The camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # Distortion coefficients 
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            _, l_rvec, l_tvec = cv2.solvePnP(leye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            _, r_rvec, r_tvec = cv2.solvePnP(reye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)


            # Get rotational matrix from rotational vector
            l_rmat, _ = cv2.Rodrigues(l_rvec)
            r_rmat, _ = cv2.Rodrigues(r_rvec)


            # [0] changes pitch
            # [1] changes roll
            # [2] changes yaw
            # +1 changes ~45 degrees (pitch down, roll tilts left (counterclockwise), yaw spins left (counterclockwise))

            # Adjust headpose vector with gaze score
            l_gaze_rvec = np.array(l_rvec)
            l_gaze_rvec[2][0] -= (lx_score-.5) * x_score_multiplier
            l_gaze_rvec[0][0] += (ly_score-.5) * y_score_multiplier

            r_gaze_rvec = np.array(r_rvec)
            r_gaze_rvec[2][0] -= (rx_score-.5) * x_score_multiplier
            r_gaze_rvec[0][0] += (ry_score-.5) * y_score_multiplier
        
            # --- Projection ---

            # Get left eye corner as integer
            l_corner = face_2d_head[2].astype(np.int32)

            # Project axis of rotation for left eye
            axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
            l_axis, _ = cv2.projectPoints(axis, l_rvec, l_tvec, cam_matrix, dist_coeffs)
            l_gaze_axis, _ = cv2.projectPoints(axis, l_gaze_rvec, l_tvec, cam_matrix, dist_coeffs)

            # Draw axis of rotation for left eye
            if draw_headpose:#False
                #if draw_full_axis:
                    #cv2.line(img, l_corner, tuple(np.ravel(l_axis[0]).astype(np.int32)), (200,200,0), 3)
                    #cv2.line(img, l_corner, tuple(np.ravel(l_axis[1]).astype(np.int32)), (0,200,0), 3)
                cv2.line(img, l_corner, tuple(np.ravel(l_axis[2]).astype(np.int32)), (0,200,200), 3)
    
            if draw_gaze:
                #if draw_full_axis:
                    #cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[0]).astype(np.int32)), (255,0,0), 3)
                    #cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[1]).astype(np.int32)), (0,255,0), 3)
                #cv2.putText(img,f"{l_gaze_axis[2]}",(30,30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                cv2.line(img, face_2d[468], tuple(np.ravel(l_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)

            
            # Get right eye corner as integer
            r_corner = face_2d_head[3].astype(np.int32)

            # Get right eye corner as integer
            r_axis, _ = cv2.projectPoints(axis, r_rvec, r_tvec, cam_matrix, dist_coeffs)
            r_gaze_axis, _ = cv2.projectPoints(axis, r_gaze_rvec, r_tvec, cam_matrix, dist_coeffs)

            # Draw axis of rotation for right eye
            if draw_headpose:#False
                #if draw_full_axis:
                    #cv2.line(img, r_corner, tuple(np.ravel(r_axis[0]).astype(np.int32)), (200,200,0), 3)
                    #cv2.line(img, r_corner, tuple(np.ravel(r_axis[1]).astype(np.int32)), (0,200,0), 3)
                cv2.line(img, r_corner, tuple(np.ravel(r_axis[2]).astype(np.int32)), (0,200,200), 3)

            if draw_gaze:
                #if draw_full_axis:
                    #cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[0]).astype(np.int32)), (255,0,0), 3)
                    #cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[1]).astype(np.int32)), (0,255,0), 3)
                cv2.line(img, face_2d[473], tuple(np.ravel(r_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)
            
            # cv2.line(img, face_2d[473], face_2d[385], (0,255,0), 1)
            # cv2.line(img, face_2d[473], face_2d[386], (0,255,0), 1)
            # cv2.line(img, face_2d[473], face_2d[380], (0,255,0), 1)
            # cv2.line(img, face_2d[473], face_2d[373], (0,255,0), 1)
            
            # cv2.line(img, face_2d[468], face_2d[160], (0,255,0), 1)
            # cv2.line(img, face_2d[468], face_2d[159], (0,255,0), 1)
            # cv2.line(img, face_2d[468], face_2d[144], (0,255,0), 1)
            # cv2.line(img, face_2d[468], face_2d[145], (0,255,0), 1)
            
            # 初始化最小距離和相應的索引
            min_distance = float('inf')
            min_distance2 = float('inf')
            min_index = None
            min_index2 = None
            face_2d_right = [face_2d[473],face_2d[385],face_2d[386],face_2d[380],face_2d[373]]
            face_2d_left  = [face_2d[468],face_2d[160],face_2d[158],face_2d[144],face_2d[145]]
            # 計算每對特徵點之間的距離
            for i in range(1, 5):
                dist = distance(face_2d[473], face_2d_right[i])
                dist2 = distance(face_2d[468], face_2d_left[i])
                # 如果找到更小的距離，更新最小距離和相應的索引
                if dist < min_distance:
                    min_distance = dist
                    min_index = face_2d_right[i]
                if dist2 < min_distance2:
                    min_distance2 = dist2
                    min_index2 = face_2d_left[i]
            
                        
            landmarks = predictor(gray, face)

            # Detect blinking
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if blinking_ratio > 5.7:
                cv2.putText(img, "Blinking", (50, 70), font, 2, (255, 0, 0))

            # Gaze detection
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

            if gaze_ratio <= 1:
                cv2.putText(img, "left", (50, 50), font, 2, (0, 0, 255), 3)
                count[0] += 1
            elif 1 < gaze_ratio < 1.7:
                cv2.putText(img, "Center", (50, 50), font, 2, (0, 0, 255), 3)
                count[1] += 1
            else:
                cv2.putText(img, "right", (50, 50), font, 2, (0, 0, 255), 3)
                count[2] += 1
        
        if frame_count % fps == 0:
            key = f"{seconds}s"
            max_count = max(count[0],count[1],count[2],)
            if max_count == count[0]:
                Record[key] = "Not Gaze"
                left_count += 1
            elif max_count == count[1]:
                Record[key] = "Gaze"
                center_count += 1
            elif max_count == count[2]:
                Record[key] = "Not Gaze"
                right_count += 1
            elif max_count == count[3]:
                Record[key] = "Not Gaze"
                out_of_range_count += 1
                
            count = [0,0,0,0]
            
            output_file = 'gaze.json'
            with open(output_file, 'w') as file:
                json.dump(Record, file, indent=4)
            seconds += 1
            result_data["Left"] = left_count
            result_data["CENTER"] = center_count
            result_data["RIGHT"] = right_count
            result_data["out_of_range"] = out_of_range_count
            result_data["seconds"] = seconds
            if seconds == 0:
                result_data["Left"] = 0
                result_data["CENTER"] = 0
                result_data["RIGHT"] = 0
            else:
                result_data["GAZED"] = round((center_count / (seconds)) * 100, 1)
                result_data["NOT GAZED"] = round(((left_count+right_count+out_of_range_count) / (seconds)) * 100, 1)
            output_file2 = 'gaze2.json'
            with open(output_file2, 'w') as file:
                json.dump(result_data, file, indent=4)  
                    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    end_time = time.time()
    duration = round((end_time - start_time),1)
    print("程式執行時間：", duration, "秒")
    cap.release()
    cv2.destroyAllWindows()
    return result_data,Record