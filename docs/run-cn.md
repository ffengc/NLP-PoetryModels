# 环境搭建和运行方法

- [简体中文](./run-cn.md)
- [English](./run-en.md)

## 环境要求

- Python 3.6.x
- TensorFlow == 1.5

## 安装环境

克隆仓库
```bash
git clone https://github.com/ffengc/NLP-PoetryModels.git
cd NLP-PoetryModels
```

创建环境
```bash
conda env create -f environment.yml
conda activate pg
pip install -r requirements.txt
```

## 训练

使用12个模型训练中文诗歌模型所需文件：`train-cn-po-all.py`

12个模型名称如下所示：

```python
valid_model_name = ['lstm', 'lstm-att', 'lstm-bi', 'lstm-att-bi',
                    'rnn', 'rnn-att', 'rnn-bi', 'rnn-att-bi',
                    'gru', 'gru-att', 'gru-bi', 'gru-att-bi']
```

训练命令(需要指定模型):

```bash
python train-cn-po-all.py --model_name lstm # 使用lstm训练
```

训练生成英文文章所需文件：`train-en-text.py`

```bash
python train-en-text.py
```

训练生成中文歌词所需文件：`train-cn-ly.py`

```bash
python train-cn-ly.py
```

训练生成代码段所需文件：`train-linux.py`

```bash
python train-linux.py
```

训练生成日语文章所需文件：`train-jpn.py`

```bash
python train-jpn.py
```

训练后的模型会保存到 `./model/` 目录下。

## 测试

使用12个模型测试中文诗歌模型所需文件：`sample-cn-po-all.py`

```bash
python sample-cn-po-all.py --model_name lstm --start_string "你好..." # 一般只需要指定模型(必选)和起始句子(可选) 其他参数可在代码中修改
```

其他测试使用：`sample.py`

命令可以参考：[https://github.com/wandouduoduo/SunRnn/blob/master/README.md](https://github.com/wandouduoduo/SunRnn/blob/master/README.md)

## BERT-CCPoem评估

参考：[https://github.com/THUNLP-AIPoet/BERT-CCPoem](https://github.com/THUNLP-AIPoet/BERT-CCPoem)