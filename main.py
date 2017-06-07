import numpy as np
import nltk.data
import tensorflow as tf
from tensorflow.contrib import learn
from flask import Flask, jsonify, render_template, request

def evaluate(input):
    vocab_path = 'mnist/data/vocab'
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(input)))

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph('mnist/data/model-2000.meta')
            checkpoint_file = tf.train.latest_checkpoint('mnist/data/')
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            #evaluate
            prediction = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})

            return prediction

# webapp
app = Flask(__name__)


@app.route('/api/evaluate', methods=['POST'])
def evaluate_api():
    input = request.form['input']
    tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
    sentences = tokenizer.tokenize(input)
    return evaluate(sentences)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
