import json
import matplotlib.pyplot as plt
import numpy as np

# 讀取 JSON 檔案
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 從各自的 JSON 檔案中讀取值 
def extract_values():
    emotion_data = load_json("emotion.json")
    
    Positive = emotion_data.get('Positive Ratio', 0)  # 假設 Head Steady 在 'Head Steady' 的鍵中
    Neutral = emotion_data.get('Neutral Ratio', 0)   # 假設 Focused On The Center 在 'Focused On The Center' 的鍵中
    Negative = emotion_data.get('Negative Ratio', 0)  # 假設 EYES OPENED 在 'EYES OPENED' 的鍵中
    
    return Positive, Neutral, Negative

# 繪製三角雷達圖
def plot_triangular_radar_chart(Positive, Neutral, Negative):
    labels = ['正面', '普通', '負面']
    values = [Positive, Neutral, Negative]

    # 計算雷達圖的角度
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    # 初始化雷達圖
    fig, ax = plt.subplots(figsize=(7, 5), subplot_kw=dict(polar=True))

    # 設置軸的背景色
    ax.set_facecolor('#faf0e6')

    # 繪製雷達圖，指定填充區域和邊框的顏色
    ax.fill(angles, values, color='#ee82ee', alpha=0.25)  # 更改填充區域的顏色
    ax.plot(angles, values, color='#ffb6c1', linewidth=2)  # 更改邊框的顏色

    # 設置雷達圖的半徑
    ax.set_ylim(0, 100)  # 根據你的數據範圍調整

    # 設置刻度標籤和標題的字型、大小和顏色
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=25, fontname='Microsoft YaHei', color='black')
    # ax.set_title('Emotion Analysis', fontsize=20, fontname='Microsoft YaHei', color='black')  # 標題
    
    # 調整刻度標籤與雷達圖的距離
    ax.tick_params(axis='x', pad=30)  # 調整刻度標籤與雷達圖的距離，可以根據需要調整 pad 的值

    # 在圖上顯示數值並加上百分號，這裡將標籤位置稍微調遠，使其離雷達圖更遠
    distance = 5  # 調整距離的值
    for angle, value, label in zip(angles, values, labels):
        ax.text(angle, value + distance, f"{value}%", ha='center', va='center', fontsize=20, fontname='Arial', color='blue')  # 使用 Arial 字型
    # 儲存為 JPG 圖片
    plt.savefig('radar_chart2.jpg')
    return
# 主程式流程
def radar():
    Positive, Neutral, Negative = extract_values()
    plot_triangular_radar_chart(Positive, Neutral, Negative)


if __name__ == '__main__':
    radar()
