# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

# 先加入注意力机制



# pick_top_n函数用于从预测的概率分布中选择可能性最高的top_n个字符
def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)  # 将输入的preds数组压缩，例如从(1,10)变为(10,)
    p[np.argsort(p)[:-top_n]] = 0  # 将概率中除了最高的top_n个值置为0
    p = p / np.sum(p)  # 将概率重新归一化，使总和为1
    c = np.random.choice(vocab_size, 1, p=p)[0]  # 根据归一化后的概率，随机选择一个字符
    return c

# CharRNN类定义了一个用于字符预测的递归神经网络模型
class CharRNN:
    # 初始化模型的参数
    def __init__(self, num_classes, num_seqs=64, num_steps=50,
                 size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128):
        if sampling is True:
            num_seqs, num_steps = 1, 1  # 如果是采样模式，批次大小和序列长度都设为1
        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        tf.reset_default_graph()  # 重置默认图
        self.build_inputs()  # 构建输入层
        self.build_lstm()    # 构建LSTM层
        self.build_loss()    # 构建损失函数
        self.build_optimizer()  # 构建优化器
        self.saver = tf.train.Saver()  # 初始化TensorFlow保存器
        print(" ---------------------- LSTM model is builded ---------------------- ")

    # 构建输入层
    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='inputs')  # 输入层
            self.targets = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='targets')  # 目标层
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout保留率

            if self.use_embedding is False:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)  # 使用one_hot编码
            else:
                with tf.device("/cpu:0"):  # 在CPU上执行嵌入层计算
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)  # 查找输入的嵌入表示

    # 构建LSTM层
    def build_lstm(self):
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)  # 创建一个基础的LSTM单元
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)  # 添加dropout层
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )  # 将多个LSTM单元堆叠成多层网络
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)  # 初始化状态为0

            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)
            # 使用dynamic_rnn展开时间维度

            # 连接和重塑输出，准备连接softmax层
            seq_output = tf.concat(self.lstm_outputs, 1)  # 连接输出
            x = tf.reshape(seq_output, [-1, self.lstm_size])  # 重塑形状为[批次大小 * 序列长度, lstm大小]

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))  # 权重
                softmax_b = tf.Variable(tf.zeros(self.num_classes))  # 偏置

            self.logits = tf.matmul(x, softmax_w) + softmax_b  # 计算logits (xW + b)
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')  # 应用softmax得到概率分布

    # 构建损失函数
    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)  # 目标值的one_hot编码
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())  # 重塑形状匹配logits
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)  # 计算交叉熵损失
            self.loss = tf.reduce_mean(loss)  # 损失的均值

    # 构建优化器
    def build_optimizer(self):
        tvars = tf.trainable_variables()  # 获取所有可训练变量
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)  # 计算梯度并进行梯度裁剪
        train_op = tf.train.AdamOptimizer(self.learning_rate)  # 使用Adam优化器
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))  # 应用梯度更新
        
    # 训练模型的方法
    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        # add log
        f = open(os.path.join(save_path, 'train.log'), 'w')
        f.write(f"step/{max_steps}, loss\n")
        self.session = tf.Session()  # 创建TensorFlow会话
        with self.session as sess:
            sess.run(tf.global_variables_initializer())  # 初始化全局变量
            step = 0  # 初始化步数
            new_state = sess.run(self.initial_state)  # 获取初始状态
            for x, y in batch_generator:  # 从生成器中循环读取数据
                step += 1  # 步数递增
                start = time.time()  # 记录开始时间
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}  # 准备输入数据
                batch_loss, new_state, _ = sess.run([self.loss, self.final_state, self.optimizer],
                                                    feed_dict=feed)  # 执行一次训练步骤

                end = time.time()  # 记录结束时间
                if step % log_every_n == 0:  # 每n步记录一次日志
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                    # 把step和loss输出到文件里面去
                    f.write(f"{step},{batch_loss}\n")
                    f.flush() # 立刻刷新缓冲区
                if (step % save_every_n == 0):  # 每n步保存一次模型
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:  # 达到最大步数时停止训练
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)  # 保存最终模型

    # 使用模型生成文本的方法
    def sample(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]  # 初始文本
        sess = self.session
        new_state = sess.run(self.initial_state)  # 获取初始状态
        preds = np.ones((vocab_size, ))  # 初始化预测概率数组
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = c  # 设置输入字符
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)  # 执行预测

        c = pick_top_n(preds, vocab_size)  # 从预测结果中选择一个字符
        samples.append(c)

        for i in range(n_samples):  # 生成更多字符
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(samples)

    # 加载模型的方法
    def load(self, checkpoint):
        self.session = tf.Session()  # 创建TensorFlow会话
        self.saver.restore(self.session, checkpoint)  # 从检查点恢复会话
        print('Restored from: {}'.format(checkpoint))
