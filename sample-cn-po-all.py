import tensorflow as tf
from read_utils import TextConverter
import os
from IPython import embed

FLAGS = tf.flags.FLAGS

'''
python sample.py \
  --use_embedding \
  --converter_path model/poetry/converter.pkl \
  --checkpoint_path model/poetry/ \
  --max_length 300
'''

# select model
tf.flags.DEFINE_string('model_name', '', 'use which model')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
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

root_path = '/root/my-SunRun/model-all-0609/'
print(f"test model: {FLAGS.model_name}")


tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_string('converter_path', root_path + f'./poetry-{FLAGS.model_name}/converter.pkl', 'converter.pkl path')
tf.flags.DEFINE_string('checkpoint_path', root_path + f'./poetry-{FLAGS.model_name}/', 'checkpoint path')
tf.flags.DEFINE_integer('max_length', 15, 'max length to generate')

tf.flags.DEFINE_integer('size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')


def main(_):
    #FLAGS.start_string = FLAGS.start_string.decode('utf-8')
    FLAGS.start_string = FLAGS.start_string
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(converter.vocab_size, sampling=True,
                    size=FLAGS.size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)

    start = converter.text_to_arr(FLAGS.start_string)
    arr = model.sample(FLAGS.max_length, start, converter.vocab_size)
    print(converter.arr_to_text(arr))


if __name__ == '__main__':
    tf.app.run()
