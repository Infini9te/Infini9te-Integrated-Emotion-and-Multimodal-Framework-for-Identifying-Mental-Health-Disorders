import cv2
from deepface import DeepFace
import numpy as np
import json
import time
import os

# Initialize counters and records
frame_count = 0
seconds = 1
RECoRD = {}
count = [0, 0, 0, 0]
result_dict = {}

# Folder path
folder_path = r'C:\Users\oscar\Desktop\UI\free kids\bad'

# Get all image files in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(('.jpg', '.jpeg', '.png'))]
start_time = time.time()

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Cannot open image: {image_file}")
        continue

    analyze = DeepFace.analyze(img, actions=['emotion'])
    emotion = analyze[0]['dominant_emotion']  # Get the dominant emotion

    # Convert emotion to positive or negative
    if emotion in ["angry", "disgust", "fear", "sad"]:
        emotion = "Negative"
        count[2] += 1
    elif emotion in ["happy", "surprise"]:
        emotion = "Positive"
        count[0] += 1
    elif emotion == "neutral":
        emotion = "Neutral"
        count[1] += 1

    key = f"{seconds} - {image_file}"
    RECoRD[key] = emotion
    
    output_file = r'C:\Users\oscar\Desktop\UI\emotion\test.json'
    with open(output_file, 'w') as file:
        json.dump(RECoRD, file, indent=4)
    seconds += 1

result_dict["Positive"] = count[0]
result_dict["Neutral"] = count[1]
result_dict["Negative"] = count[2]
result_dict["Unknown"] = count[3]
result_dict["Total"] = seconds - 1
result_dict["Positive Ratio"] = round((count[0] / (seconds - 1)) * 100, 1)
result_dict["Neutral Ratio"] = round((count[1] / (seconds - 1)) * 100, 1)
result_dict["Negative Ratio"] = round((count[2] / (seconds - 1)) * 100, 1)
result_dict["Unknown Ratio"] = round((count[3] / (seconds - 1)) * 100, 1)

# Write statistics to JSON file
output_file2 = r'C:\Users\oscar\Desktop\UI\emotion\test2.json'
with open(output_file2, 'w') as file:
    json.dump(result_dict, file, indent=4)

end_time = time.time()
total_time = end_time - start_time
print("Spend time:", total_time, "s")
