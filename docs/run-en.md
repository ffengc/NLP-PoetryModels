# Environment Setup and Running Methods

- [Simplified Chinese](./run-cn.md)
- [English](./run-en.md)

## Environment Requirements

- Python 3.6.x
- TensorFlow == 1.5

## Setting Up the Environment

Clone the repository:
```bash
git clone https://github.com/ffengc/NLP-PoetryModels.git
cd NLP-PoetryModels
```

Create the environment:
```bash
conda env create -f environment.yml
conda activate pg
pip install -r requirements.txt
```

## Training

Use the following file to train the Chinese poetry model with 12 models: `train-cn-po-all.py`

The 12 model names are as follows:

```python
valid_model_name = ['lstm', 'lstm-att', 'lstm-bi', 'lstm-att-bi',
                    'rnn', 'rnn-att', 'rnn-bi', 'rnn-att-bi',
                    'gru', 'gru-att', 'gru-bi', 'gru-att-bi']
```

Training command (specify model required):

```bash
python train-cn-po-all.py --model_name lstm # Train using lstm
```

Use the following file to train for generating English articles: `train-en-text.py`

```bash
python train-en-text.py
```

Use the following file to train for generating Chinese song lyrics: `train-cn-ly.py`

```bash
python train-cn-ly.py
```

Use the following file to train for generating code snippets: `train-linux.py`

```bash
python train-linux.py
```

Use the following file to train for generating Japanese articles: `train-jpn.py`

```bash
python train-jpn.py
```

Trained models will be saved in the `./model/` directory.

## Testing

Use the following file to test the Chinese poetry model with 12 models: `sample-cn-po-all.py`

```bash
python sample-cn-po-all.py --model_name lstm --start_string "你好..." # Generally, only the model (mandatory) and starting sentence (optional) need to be specified; other parameters can be modified in the code
```

For other tests, use: `sample.py`

Refer to commands at: [https://github.com/wandouduoduo/SunRnn/blob/master/README.md](https://github.com/wandouduoduo/SunRnn/blob/master/README.md)

## BERT-CCPoem Assessment

Refer to: [https://github.com/THUNLP-AIPoet/BERT-CCPoem](https://github.com/THUNLP-AIPoet/BERT-CCPoem)