# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

class CharRNN_GRU:
    def __init__(self, num_classes, num_seqs=64, num_steps=50,
                 size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128):
        if sampling is True:
            num_seqs, num_steps = 1, 1
        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.gru_size = size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self.build_inputs()
        self.build_gru()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()
        print(" ---------------------- GRU_ATTENTION model is builded ---------------------- ")

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            if self.use_embedding is False:
                self.gru_inputs = tf.one_hot(self.inputs, self.num_classes)
            else:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    self.gru_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
    
    
    # 增加注意力机制    
    def attention_layer(self, inputs, attention_size):
        hidden_size = inputs.shape[2].value  # LSTM的隐藏层大小
        W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('v'):
            # 对LSTM输出应用tanh非线性变换，并计算“对齐”分数
            v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # vu的形状为 [batch_size, sequence_length]
        alphas = tf.nn.softmax(vu, name='alphas')  # softmax结果，得到权重，形状为 [batch_size, sequence_length]
        # 保持输出形状为 (32, 26, 128)，通过扩展alphas维度并进行元素乘法
        alphas_expanded = tf.expand_dims(alphas, -1)  # 扩展维度至 [batch_size, sequence_length, 1]
        # 加权和，得到上下文向量
        # output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        output = inputs * alphas_expanded
        return output, alphas_expanded

    def build_gru(self):
        def get_a_cell(gru_size, keep_prob):
            gru = tf.nn.rnn_cell.GRUCell(gru_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(gru, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('gru'):
            cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.gru_size, self.keep_prob) for _ in range(self.num_layers)])
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)
            self.gru_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.gru_inputs, initial_state=self.initial_state)
            
            # 将注意力机制应用于LSTM输出
            with tf.name_scope("Attention_layer"):
                self.attention_output, self.alphas = self.attention_layer(self.gru_outputs, self.gru_size)
                
            x = tf.reshape(self.attention_output, [-1, self.gru_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.gru_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        f = open(os.path.join(save_path, 'train.log'), 'w')
        f.write(f"step/{max_steps}, loss\n")
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x, self.targets: y, self.keep_prob: self.train_keep_prob, self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss, self.final_state, self.optimizer], feed_dict=feed)
                end = time.time()
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                    f.write(f"{step},{batch_loss}\n")
                    f.flush()
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def sample(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x, self.keep_prob: 1., self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state], feed_dict=feed)
        c = pick_top_n(preds, vocab_size)
        samples.append(c)
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x, self.keep_prob: 1., self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state], feed_dict=feed)
            c = pick_top_n(preds, vocab_size)
            samples.append(c)
        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
