import tensorflow as tf
from read_utils import TextConverter, batch_generator
import os
import codecs

FLAGS = tf.flags.FLAGS

# ----------------------------- pdb ----------------------------- #
import sys, os, pdb
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
# ----------------------------- pdb ----------------------------- #

'''
python train.py \
  --use_embedding \
  --input_file data/poetry.txt \
  --name poetry \
  --learning_rate 0.005 \
  --num_steps 26 \
  --num_seqs 32 \
  --max_steps 10000
'''

# select model
tf.flags.DEFINE_string('model_name', '', 'use which model')
valid_model_name = ['lstm', 'lstm-att', 'lstm-bi', 'lstm-att-bi',
                    'rnn', 'rnn-att', 'rnn-bi', 'rnn-att-bi',
                    'gru', 'gru-att', 'gru-bi', 'gru-att-bi']
def usage():
    print("unvalid model" + "\n" + "\t" + "valid model name: " + str(valid_model_name))
    exit()
if FLAGS.model_name == '' or FLAGS.model_name not in valid_model_name:
    usage()
# lstm
elif FLAGS.model_name == 'lstm':
    from config.model import CharRNN
elif FLAGS.model_name == 'lstm-att':
    from config.model_atte import CharRNN
elif FLAGS.model_name == 'lstm-bi':
    from config.model_bi import CharRNN
elif FLAGS.model_name == 'lstm-att-bi':
    from config.model_att_bi import CharRNN
# rnn
elif FLAGS.model_name == 'rnn':
    from config.model_rnn import CharRNN_RNN as CharRNN
elif FLAGS.model_name == 'rnn-att':
    from config.model_rnn_atte import CharRNN_RNN as CharRNN
elif FLAGS.model_name == 'rnn-bi':
    from config.model_rnn_bi import CharRNN_RNN as CharRNN
elif FLAGS.model_name == 'rnn-att-bi':
    from config.model_rnn_att_bi import CharRNN_RNN as CharRNN
# gru
elif FLAGS.model_name == 'gru':
    from config.model_gru import CharRNN_GRU as CharRNN
elif FLAGS.model_name == 'gru-att':
    from config.model_gru_atte import CharRNN_GRU as CharRNN
elif FLAGS.model_name == 'gru-bi':
    from config.model_gru_bi import CharRNN_GRU as CharRNN
elif FLAGS.model_name == 'gru-att-bi':
    from config.model_gru_att_bi import CharRNN_GRU as CharRNN


# input
root_path = "/root/my-SunRun/"
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_string('input_file', root_path + 'data/poetry.txt', 'utf8 encoded text file')
tf.flags.DEFINE_string('name', 'poetry', 'name of the model')
tf.flags.DEFINE_float('learning_rate', 0.005, 'learning_rate')
tf.flags.DEFINE_integer('num_seqs', 32, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 26, 'length of one seq')
tf.flags.DEFINE_integer('max_steps', 10000, 'max steps to train')
# default
tf.flags.DEFINE_integer('size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')


def main(_):
    model_path = os.path.join('model', FLAGS.name + '-' + FLAGS.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter(text, FLAGS.max_vocab)
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)
    print(converter.vocab_size)
    # ForkedPdb().set_trace()
    model = CharRNN(converter.vocab_size,
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    size=FLAGS.size, # BUG
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.train(g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()