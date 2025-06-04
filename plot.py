import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('/home/quan/yolo11_tea/runs/detect/train3/results.csv')

# 设置图表大小
plt.figure(figsize=(16, 10))

# 绘制训练损失曲线
plt.subplot(2, 2, 1)
plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss')
plt.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 绘制精度和召回率曲线
plt.subplot(2, 2, 2)
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
plt.title('Precision and Recall per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

# 绘制mAP曲线
plt.subplot(2, 2, 3)
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@.50')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@.50-.95')
plt.title('mAP per Epoch')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.legend()
plt.grid(True)

# 绘制学习率曲线
plt.subplot(2, 2, 4)
plt.plot(df['epoch'], df['lr/pg0'], label='Learning Rate pg0')
plt.plot(df['epoch'], df['lr/pg1'], label='Learning Rate pg1')
plt.plot(df['epoch'], df['lr/pg2'], label='Learning Rate pg2')
plt.title('Learning Rate per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid(True)

# 自动调整子图间距
plt.tight_layout()
plt.show()
# 保存图像到本地
plt.savefig('./assets/training_curves.png', dpi=300, bbox_inches='tight')  # dpi可调清晰度，bbox_inches防止裁剪边缘

# 可选：关闭绘图以释放内存
plt.close()