import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib import rcParams



# 加载数据
def load_data(file_name):
    return pd.read_csv(file_name, sep=" ", names=["step", "train_loss"])


# # 设置全局字体为Times New Roman
# rcParams['font.family'] = 'Times New Roman'

# 绘制每个模型的损失曲线
def plot_loss(data, label, color):
    plt.plot(data[:,0].tolist(), data[:,1].tolist(), label=label, color=color)

# 文件路径，这里假设你的文件名按照下面的格式
file_names = [f"/root/my-SunRun/model-all-0609/poetry/train.log", 
              f"/root/my-SunRun/model-all-0609/poetry-rnn/train.log", 
              f"/root/my-SunRun/model-all-0609/poetry-gru/train.log"]
labels = ["LSTM", "RNN", "GRU"]
colors = ['blue', 'green', 'red']

plt.figure(figsize=(10, 6))

# 循环读取文件并绘制图形
for file_name, label, color in zip(file_names, labels, colors):
    arr = pd.read_csv(file_name).to_numpy()
    plot_loss(arr, label, color)

# 添加图例和标题
plt.legend()
plt.title('Diff BaseModel Loss Comparison')
plt.xlabel('Step')
plt.ylabel('Train Loss')
plt.grid(True)
plt.savefig(f"comp.png", dpi=1000)
print("done")