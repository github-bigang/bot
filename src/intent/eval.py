#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
from utils.vocab_utils import Vocab

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_path", "./data/intent/small/", "Path for the intent data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 2, "Batch Size (default: 64)")
#1511626669 for no char originally
#1511667468 for char
#1511666657 for no char
tf.flags.DEFINE_string("checkpoint_dir", "runs/1511667468/checkpoints/", "Checkpoint directory from training run") # 
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
word_vocab = Vocab('resources/w2v_cn_wiki_100.txt', fileformat='txt3')
# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
if FLAGS.eval_train:
    x_raw, y_test, x_chars, max_document_length, max_document_char_length, char_vocab_size = \
        data_helpers.load_data_and_labels(word_vocab, FLAGS.data_path + '/intent-seg.train', vocab_processor=vocab_processor)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw, y_test, x_chars, max_document_length, max_document_char_length, char_vocab_size = \
        data_helpers.load_data_and_labels(word_vocab, FLAGS.data_path + '/intent-seg.test', vocab_processor=vocab_processor)
    y_test = np.argmax(y_test, axis=1)

max_document_length, max_document_char_length = 17, 27
# x_test = np.array(list(vocab_processor.transform(x_raw)))
x_test, x_char = [], []
for i in range(len(x_raw)):
    arr = [0] * max_document_length
    arr[:len(x_raw[i])] = x_raw[i]
    x_test.append(arr)
    arr = [0] * max_document_char_length
    arr[:len(x_chars[i])] = x_chars[i]
    x_char.append(arr)
x_test=np.array(x_test)
x_char=np.array(x_char)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_x_char = graph.get_operation_by_name("input_x_char").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        x_input = [(x_test[i], x_char[i]) for i in range(len(x_test))]
        batches = data_helpers.batch_iter(list(x_input), FLAGS.batch_size, 1, shuffle=False)
#         batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
#         batches_char = data_helpers.batch_iter(list(x_char), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

#         for i in range(len(batches)):
        for x_test_batch in batches:
#             x_test_batch, x_char_batch = batches[i], batches_char[i]
            batch_predictions = sess.run(predictions, {input_x: [x[0] for x in x_test_batch], input_x_char: [x[1] for x in x_test_batch], dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)