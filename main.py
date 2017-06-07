import numpy as np
import os
import nltk.data
import tensorflow as tf
from tensorflow.contrib import learn
from flask import Flask, jsonify, render_template, request

sess = None
predictions = None
input_x = None
dropout_keep_prob = None


# Evaluation Method
def evaluate(input):
    # evaluate
    vocab_path = os.path.join(app.root_path, 'mnist','data','vocab')
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(input)))

    prediction = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})
    return prediction.tolist()


# webapp
app = Flask(__name__)
print()

# set up the model before the app accecpts requests.
@app.before_first_request
def _run_on_start():
    graph = tf.Graph()
    with graph.as_default():
        global sess
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            checkpoint_file = tf.train.latest_checkpoint(os.path.join(app.root_path, 'mnist','data'))
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            global input_x
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            global dropout_keep_prob
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            global predictions
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]


@app.route('/api/evaluate', methods=['POST'])
def evaluate_api():
    input = request.form['input']
    tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
    sentences = tokenizer.tokenize(input)
    sentences_to_lower = [x.lower() for x in sentences]
    results = evaluate(sentences_to_lower)
    output = []
    for idx, val in enumerate(results):
        if val == 0:
            output.append([sentences[idx], 'negative'])
        if val == 1:
            output.append([sentences[idx], 'positive'])

    return render_template('index.html', original_text=input, output=output)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

