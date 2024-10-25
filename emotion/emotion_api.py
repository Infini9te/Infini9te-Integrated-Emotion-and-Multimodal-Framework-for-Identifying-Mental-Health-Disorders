from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import cv2
import os
import mediapipe as mp
import time
from tensorflow import keras
from keras.applications.resnet import preprocess_input
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

def load_model(model_json_path, model_weights_path):
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_path)
    return loaded_model

# 加載模型
script_dir = os.path.dirname(os.path.abspath(__file__))
model_json_path = os.path.join(script_dir, 'model_json.json')
model_weights_path = os.path.join(script_dir, 'model_weights.h5')
model = load_model(model_json_path, model_weights_path)

# 確保存在 uploads 目錄
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Emotion Analysis</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        h1 {
            color: #444;
            text-align: center;
        }
        form {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
        input[type="file"] {
            margin-right: 10px;
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 5px;
        }
        button {
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            background-color: #5cb85c;
            color: white;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #4cae4c;
        }
        pre {
            background: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        img {
            margin-top: 20px;
            display: block;
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .button-group {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .button-group button {
            margin: 0 10px;
        }
    </style>
</head>
<body>
    <h1>影片情緒分析</h1>
    <form id="uploadForm">
        <input type="file" id="videoFile" name="video" accept="video/*" required>
        <button type="submit"><i class="fas fa-upload"></i> 上傳並分析</button>
    </form>
    <pre id="result"></pre>
    <h2>情緒統計</h2>
    <div id="emotionStats" style="display:none;"></div>
    <h2>情緒雷達圖</h2>
    <img id="emotionRadar" src="" alt="Emotion Radar" style="display:none;">
    <h2>每秒情緒狀態</h2>
    <img id="timeSeries" src="" alt="Time Series" style="display:none;">
    
    <div class="button-group" style="display:none;">
        <button id="downloadRadar"><i class="fas fa-download"></i> 下載情緒雷達圖</button>
        <button id="downloadTimeSeries"><i class="fas fa-download"></i> 下載時序圖</button>
        <button id="downloadVideo"><i class="fas fa-download"></i> 下載影片</button>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const fileInput = document.getElementById('videoFile');
            const formData = new FormData();
            formData.append('video', fileInput.files[0]);
            
            const response = await fetch('/process_video', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();

            const stats = `
                正面情緒數: ${result.Positive} (${result['Positive Ratio']}%)
                中立情緒數: ${result.Neutral} (${result['Neutral Ratio']}%)
                負面情緒數: ${result.Negative} (${result['Negative Ratio']}%)
                不明情緒數: ${result.Unknown}
                總時間: ${result.Total}秒
            `;
            document.getElementById('emotionStats').innerText = stats;
            document.getElementById('emotionStats').style.display = 'block';

            document.getElementById('emotionRadar').src = result.emotionRadar;
            document.getElementById('emotionRadar').style.display = 'block';
            document.getElementById('timeSeries').src = result.timeSeries;
            document.getElementById('timeSeries').style.display = 'block';

            const videoID = result.Video_ID;
            const outputVideoName = result.output_video_name;  // 取得影片檔名

            document.querySelector('.button-group').style.display = 'flex';

            document.getElementById('downloadRadar').onclick = () => {
                const radarData = result.emotionRadar.split(',')[1];
                const link = document.createElement('a');
                link.href = `data:image/png;base64,${radarData}`;
                link.download = 'emotion_radar.png';
                link.click();
            };

            document.getElementById('downloadTimeSeries').onclick = () => {
                const timeSeriesData = result.timeSeries.split(',')[1];
                const link = document.createElement('a');
                link.href = `data:image/png;base64,${timeSeriesData}`;
                link.download = 'time_series.png';
                link.click();
            };

            document.getElementById('downloadVideo').onclick = () => {
            const link = document.createElement('a');
            link.href = `/uploads/${outputVideoName}`;
            link.download = outputVideoName;
            link.click();
            };
        });
    </script>
</body>
</html>
    ''')

@app.route('/process_video', methods=['POST'])
def process_video():
    start_time = time.time()
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    video_path = os.path.join('uploads', video_file.filename)
    video_file.save(video_path)

    video_capture = cv2.VideoCapture(video_path)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0]

    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    json_name = "uploads/emotion.json"
    open(json_name, "w").close()
    json_name2 = "uploads/time.json"
    open(json_name2, "w").close()

    positive_count = 0
    negative_count = 0
    neutral_count = 0
    frame_count = 0
    seconds = 0
    Count = [0, 0, 0, 0]
    emotion_per_second = {}
    result_dict = {}
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 2 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_detection = face_detection.process(frame_rgb)

            if results_detection.detections:
                for detection in results_detection.detections:
                    results_mesh = face_mesh.process(frame_rgb)

                    if results_mesh.multi_face_landmarks:
                        processed_frame = cv2.resize(frame, (48, 48))
                        processed_frame = np.expand_dims(processed_frame, axis=0)
                        processed_frame = preprocess_input(processed_frame)

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
            if negative_count == 0 and neutral_count == 0 and positive_count == 0:
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
    result_dict["Positive Ratio"] = round((Count[0] / seconds) * 100, 1)
    result_dict["Neutral Ratio"] = round((Count[1] / seconds) * 100, 1)
    result_dict["Negative Ratio"] = round((Count[2] / seconds) * 100, 1)

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y/%m/%d %H:%M:%S")

    result_dict["Execution Datetime"] = formatted_datetime

    with open(json_name, "w") as file:
        json.dump(result_dict, file, indent=4)

    video_capture.release()
    
    with open(json_name2, "r") as file:
        emotion_data = json.load(file)

    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_name = os.path.join('uploads', f"{video_id}_output.mp4")
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
                        text_color = (0, 255, 0)
                    elif emotion_text == 'Negative':
                        text_color = (0, 0, 255)
                    elif emotion_text == 'Neutral':
                        text_color = (255, 0, 0)
                    elif emotion_text == 'Unknown':
                        text_color = (0, 0, 0)
                        
                    cv2.rectangle(frame, (x, y), (x + w, y + h), text_color, 2)
                    cv2.putText(frame, f"Emotion: {emotion_text}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2) 
                    
        out.write(frame)

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
        
    end_time = time.time()
    total_time = end_time - start_time
    print("Spend time:", total_time, "s")

    # 生成雷達圖
    emotion_radar = plot_emotion_radar(result_dict)

    # 生成時序圖
    time_series = plot_time_series(emotion_per_second)

    result_dict["emotionRadar"] = emotion_radar
    result_dict["timeSeries"] = time_series

    return jsonify({
        "Positive": Count[0],
        "Neutral": Count[1],
        "Negative": Count[2],
        "Unknown": Count[3],
        "Total": seconds,
        "Positive Ratio": round((Count[0] / seconds) * 100, 1),
        "Neutral Ratio": round((Count[1] / seconds) * 100, 1),
        "Negative Ratio": round((Count[2] / seconds) * 100, 1),
        "emotionRadar": emotion_radar,
        "timeSeries": time_series,
        "Video_ID": video_id,
        "output_video_name": f"{video_id}_output.mp4"  # 傳回影片檔名
    }), 200

def plot_emotion_radar(emotion_counts):
    labels = ['Neutral', 'Positive', 'Negative']
    values = [emotion_counts['Positive'], emotion_counts['Neutral'], emotion_counts['Negative']]

    # 計算雷達圖的角度
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    # 初始化雷達圖
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))


    # 設置軸的背景色
    ax.set_facecolor('#faf0e6')

    # 繪製雷達圖，指定填充區域和邊框的顏色
    ax.fill(angles, values, color='#ee82ee', alpha=0.25)  # 更改填充區域的顏色
    ax.plot(angles, values, color='#ffb6c1', linewidth=2)  # 更改邊框的顏色

    # 設置雷達圖的半徑
    ax.set_ylim(0, max(values) + 10)  # 根據數據範圍調整

    # 設置刻度標籤和標題的字型、大小和顏色
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontname='Arial', color='black')
    ax.set_title('Emotion Analysis', fontsize=20, fontname='Arial', color='black')  # 使用 Arial 字型
    
    # 調整刻度標籤與雷達圖的距離
    ax.tick_params(axis='x', pad=30)  # 調整刻度標籤與雷達圖的距離，可以根據需要調整 pad 的值

    # 在圖上顯示數值並加上百分號，這裡將標籤位置稍微調遠，使其離雷達圖更遠
    distance = 5  # 調整距離的值
    for angle, value, label in zip(angles, values, labels):
        ax.text(angle, value, f"{value}%", ha='center', va='center', fontsize=10, fontname='Arial', color='blue')


    # buf = BytesIO()
    # plt.savefig(buf, format='png')
    # plt.close(fig)
    # buf.seek(0)

     # 儲存圖片到後端
    radar_path = os.path.join('uploads', 'emotion_radar.png')
    plt.savefig(radar_path, format='png', bbox_inches='tight')
    plt.close(fig)

    return "data:image/png;base64," + base64.b64encode(open(radar_path, 'rb').read()).decode()


def plot_time_series(emotion_per_second):
     # 将情绪数据转换为数值数据
    time_slots = list(emotion_per_second.keys())
    emotions = list(emotion_per_second.values())
    emotion_colors = {"Positive": "skyblue", "Neutral": "lightgreen", "Negative": "salmon"}

    # 设置图形和轴，调整大小为(15, 6)
    fig, ax = plt.subplots(figsize=(10, 0.65))
    y = [0]  # 所有条形图的 y 位置

    # 创建图例
    legend_colors = [plt.Rectangle((0,0),1,1, color=color) for color in emotion_colors.values()]
    legend_labels = list(emotion_colors.keys())

    # 绘制每个时间段的情绪
    for i, (time, emotion) in enumerate(emotion_per_second.items()):
        ax.barh(0, 1, 1, left=i, color=emotion_colors[emotion], edgecolor='black', linewidth=0.5)

    # 设置 x 轴刻度和标签
    ax.set_xticks(range(len(time_slots)))
    ax.set_xticklabels(time_slots, rotation=0)
    ax.set_yticks(y)
    ax.set_yticklabels(["Emotion"], fontsize=12)

    # 设置底部和左侧的坐标轴属性
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_visible(True)
    # ax.spines['left'].set_bounds((-1, 1))
    # ax.spines['bottom'].set_bounds((0, len(time_slots)))

    # 添加图例
    ax.legend(legend_colors, legend_labels,bbox_to_anchor=(1, -0.32))

    # 添加标签和标题
    ax.set_xlabel('Time(s)', fontsize=14, color="black")
    ax.set_title('Emotion Over Time', fontsize=16)

    ax.set_ylim(-2, 2)  # 控制 Y 轴范围
    
    # buf = BytesIO()
    # plt.savefig(buf, format='png', bbox_inches='tight')
    # plt.close()
    # buf.seek(0)

    # 儲存圖片到後端
    time_series_path = os.path.join('uploads', 'time_series.png')
    plt.savefig(time_series_path, format='png', bbox_inches='tight')
    plt.close()

    return "data:image/png;base64," + base64.b64encode(open(time_series_path, 'rb').read()).decode()


@app.route('/uploads/<path:filename>', methods=['GET'])
def download_file(filename):


    return send_from_directory('uploads', filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
