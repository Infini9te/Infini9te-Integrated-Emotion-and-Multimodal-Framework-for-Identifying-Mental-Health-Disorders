import cv2
from deepface import DeepFace
import json
import time
import os

def analyze_and_annotate_emotion(image_path, output_folder):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot open image: {image_path}")
        return None

    try:
        analyze = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        for face in analyze:
            emotion = face['dominant_emotion']
            bbox = face['region']
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

            if emotion in ["angry", "disgust", "fear", "sad"]:
                emotion = "Negative"
            elif emotion in ["happy", "surprise"]:
                emotion = "Positive"

            # Draw bounding box and emotion text
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save annotated image
        annotated_image_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(annotated_image_path, img)

        return emotion

    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return None

def save_emotions_to_json(record, output_file):
    with open(output_file, 'w') as file:
        json.dump(record, file, indent=4)

def save_emotion_statistics(emotion_counts, output_file):
    total = sum(emotion_counts.values())
    emotion_statistics = {emotion: {"count": count, "proportion": count / total} for emotion, count in emotion_counts.items()}
    
    with open(output_file, 'w') as file:
        json.dump(emotion_statistics, file, indent=4)

def main(folder_path, output_folder, output_file, stats_output_file):
    frame_count = 0
    seconds = 0
    RECoRD = {}
    emotion_counts = {}

    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(('.jpg', '.jpeg', '.png'))]
    start_time = time.time()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        emotion = analyze_and_annotate_emotion(image_path, output_folder)
        
        if emotion:
            key = f"{seconds}"
            RECoRD[key] = emotion
            save_emotions_to_json(RECoRD, output_file)

            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
            
            seconds += 1

    save_emotion_statistics(emotion_counts, stats_output_file)

    end_time = time.time()
    total_time = end_time - start_time
    print("Spend time:", total_time, "s")

if __name__ == "__main__":
    folder_path = 'C:\\Users\\z3412\\Desktop\\b'   #需要驗證的資料夾  
    output_folder = 'C:\\Users\\z3412\\Desktop'   #輸出資料夾
    output_file = 'C:\\Users\\z3412\\Desktop\\emotion.json' #各情緒圖片情緒
    stats_output_file = 'C:\\Users\\z3412\\Desktop\\summary.json'    #各情緒總數及比例
    main(folder_path, output_folder, output_file, stats_output_file)
