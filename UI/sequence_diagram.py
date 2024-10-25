import json
import matplotlib.pyplot as plt

# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():  # 从 JSON 文件中提取数据
    data = load_json('time.json')

    # 将情绪数据转换为数值数据
    time_slots = list(data.keys())
    emotions = list(data.values())
    emotion_colors = {"Positive": "skyblue", "Neutral": "lightgreen", "Negative": "salmon"}

    # 根据时间段的数量动态设置图形的宽度
    num_time_slots = len(time_slots)
    fig_width = max(10, num_time_slots * 1)  # 设置宽度以适应刻度标签，最小宽度为10
    fig_height = 0.56  # 固定高度

    # 设置图形和轴
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    y = [0]  # 所有条形图的 y 位置

    # 创建图例
    legend_colors = [plt.Rectangle((0,0),1,1, color=color) for color in emotion_colors.values()]
    legend_labels = list(emotion_colors.keys())

    # 绘制每个时间段的情绪
    for i, (time, emotion) in enumerate(data.items()):
        ax.barh(0, 1, 1, left=i, color=emotion_colors[emotion], edgecolor='black', linewidth=0.5)

    # 设置 x 轴刻度和标签，每秒都显示
    ax.set_xticks(range(len(time_slots)))
    ax.set_xticklabels(time_slots)  # 保持标签水平

    ax.set_yticks(y)
    ax.set_yticklabels(["Emotion"], fontsize=12)

    # 添加图例
    ax.legend(legend_colors, legend_labels, bbox_to_anchor=(1, 0))

    # 添加标签和标题
    ax.set_xlabel('Time(s)', fontsize=14, color="black")
    ax.set_title('Emotion Sequence Diagram', fontsize=16)

    ax.set_ylim(-2, 2)  # 控制 Y 轴范围

    # 调整布局并显示图形
    plt.savefig('abc.jpg', bbox_inches='tight')

if __name__ == '__main__':
    main()

