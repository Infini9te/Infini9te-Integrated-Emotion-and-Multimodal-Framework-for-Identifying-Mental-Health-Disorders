import cv2
import os
import mediapipe as mp
import time
from tensorflow import keras
from keras.applications.resnet import preprocess_input
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_json_path, model_weights_path):
    try:
        with open(model_json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights_path)
        logging.info("Model loaded successfully.")
        return loaded_model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def resize_image(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image

def detect_and_classify_emotions(image_files, model, folder_path, output_folder, face_detection, emotion_map, color_map):
    emotion_per_image = {}
    emotion_count = {emotion: 0 for emotion in list(emotion_map.values()) + ["Unknown"]}

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Cannot open image: {image_file}")
            continue

        # 調整解析度
        img = resize_image(img, 800, 600)

        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results_detection = face_detection.process(frame_rgb)
        detected_emotion = "Unknown"

        if results_detection.detections:
            for detection in results_detection.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                x2, y2 = x1 + int(bboxC.width * iw), y1 + int(bboxC.height * ih)

                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                processed_frame = cv2.resize(face, (48, 48))
                processed_frame = np.expand_dims(processed_frame, axis=0)
                processed_frame = preprocess_input(processed_frame)

                prediction = model.predict(processed_frame)
                emotion_label = np.argmax(prediction)
                detected_emotion = emotion_map.get(emotion_label, "Unknown")

                color = color_map.get(detected_emotion, (128, 128, 128))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, detected_emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        emotion_per_image[image_file] = detected_emotion
        emotion_count[detected_emotion] += 1

        output_image_path = os.path.join(output_folder, f"output_{image_file}")
        cv2.imwrite(output_image_path, img)

    return emotion_per_image, emotion_count

def save_results(emotion_per_image, emotion_count, json_output_path, summary_output_path):
    with open(json_output_path, "w") as file:
        json.dump(emotion_per_image, file, indent=4)

    total_images = sum(emotion_count.values())
    emotion_ratio = {key: round((value / total_images) * 100, 1) for key, value in emotion_count.items()}

    result_dict = {
        "Emotion Count": emotion_count,
        "Emotion Ratio": emotion_ratio,
        "Total Images": total_images
    }

    with open(summary_output_path, "w") as file:
        json.dump(result_dict, file, indent=4)

def main():
    start_time = time.time()

    folder_path = r'C:\Users\z3412\Desktop\list'    #需要辨識的資料夾
    output_folder = r'C:\Users\z3412\Desktop\my\800x600\all'    #輸出資料夾
    model_json_path = r'C:\Users\z3412\Desktop\emotion\new code\0703\model_json.json'   #資料庫 model_json.json
    model_weights_path = r'C:\Users\z3412\Desktop\emotion\new code\0703\model_weights.h5'   #資料庫 model_weights.h5
    json_output_path = r'C:\Users\z3412\Desktop\my\800x600\all\emotion.json'  #各圖片的情緒
    summary_output_path = r'C:\Users\z3412\Desktop\my\800x600\all\summary.json'   #各情緒總數及比例

    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(('.jpg', '.jpeg', '.png'))]

    model = load_model(model_json_path, model_weights_path)
    if model is None:
        logging.error("Failed to load model. Exiting.")
        return

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    emotion_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    color_map = {
        "Positive": (0, 255, 0),
        "Neutral": (255, 0, 0),
        "Negative": (0, 0, 255),
        "Unknown": (128, 128, 128)
    }

    emotion_per_image, emotion_count = detect_and_classify_emotions(image_files, model, folder_path, output_folder, face_detection, emotion_map, color_map)

    save_results(emotion_per_image, emotion_count, json_output_path, summary_output_path)

    cv2.destroyAllWindows()

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Elapsed time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
