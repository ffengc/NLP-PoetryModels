import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib import rcParams



basic_model_name = "GRU"

if basic_model_name == "LSTM":
    save_name = "poetry"
elif basic_model_name == "GRU":
    save_name = "poetry-gru"
elif basic_model_name == "RNN":
    save_name = "poetry-rnn"
else:
    assert(False)

# 加载数据
def load_data(file_name):
    return pd.read_csv(file_name, sep=" ", names=["step", "train_loss"])


# # 设置全局字体为Times New Roman
# rcParams['font.family'] = 'Times New Roman'

# 绘制每个模型的损失曲线
def plot_loss(data, label, color):
    plt.plot(data[:,0].tolist(), data[:,1].tolist(), label=label, color=color)

# 文件路径，这里假设你的文件名按照下面的格式
file_names = [f"/root/my-SunRun/model-all-0609/{save_name}/train.log", 
              f"/root/my-SunRun/model-all-0609/{save_name}-att/train.log", 
              f"/root/my-SunRun/model-all-0609/{save_name}-bi/train.log", 
              f"/root/my-SunRun/model-all-0609/{save_name}-att-bi/train.log"]
labels = [basic_model_name, f"{basic_model_name} + Attention", f"Bi-{basic_model_name}", f"Bi-{basic_model_name} + Attention"]
colors = ['blue', 'green', 'red', 'purple']

plt.figure(figsize=(10, 6))

# 循环读取文件并绘制图形
for file_name, label, color in zip(file_names, labels, colors):
    arr = pd.read_csv(file_name).to_numpy()
    plot_loss(arr, label, color)

# 添加图例和标题
plt.legend()
plt.title('Training Loss Comparison')
plt.xlabel('Step')
plt.ylabel('Train Loss')
plt.grid(True)
plt.savefig(f"{basic_model_name}.png", dpi=1000)
print("done")