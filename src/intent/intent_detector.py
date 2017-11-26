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

class intent_detector:
    def __init__(self, model_path='runs/1511667468/checkpoints/'):
        #1511667468 for char
        #1511666657 for no char
        # Parameters
        self.model_path = model_path
#         tf.flags.DEFINE_string("data_path", "./data/intent/small/", "Path for the intent data.")
#         
#         # Eval Parameters
#         tf.flags.DEFINE_integer("batch_size", 2, "Batch Size (default: 64)")
# #         tf.flags.DEFINE_string("checkpoint_dir", "runs/1511626669/checkpoints/", "Checkpoint directory from training run")
#         tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
#         
#         # Misc Parameters
#         tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
#         tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#         
#         
#         FLAGS = tf.flags.FLAGS
#         FLAGS._parse_flags()
#         print("\nParameters:")
#         for attr, value in sorted(FLAGS.__flags.items()):
#             print("{}={}".format(attr.upper(), value))
#         print("")

        self.word_vocab = Vocab('resources/w2v_cn_wiki_100.txt', fileformat='txt3')
        self.max_document_length, self.max_document_char_length = 17, 27
        vocab_path = os.path.join(self.model_path, "..", "vocab")
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        
        checkpoint_file = tf.train.latest_checkpoint(self.model_path)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=True,
              log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
        
                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("input_x").outputs[0]
                self.input_x_char = graph.get_operation_by_name("input_x_char").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        
                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]

    def detect(self, sentence, label = None, save_path=None):
        x_raw = [self.word_vocab.to_index_sequence(' '.join([x.decode('utf-8') for x in sentence.split(' ')]))]
        all_chars = [''.join([x + ' ' for x in sentence.replace(' ', '').decode('utf-8')])]
        x_chars = np.array(list(self.vocab_processor.fit_transform(all_chars)))
        y_test = None
        if label: 
            y_test = [0, 0, 0]
            y_test[int(label) - 1] = 1

        # Map data into vocabulary
#         vocab_path = os.path.join(self.model_path, "..", "vocab")
        x_test, x_char = [], []
        for i in range(len(x_raw)):
            arr = [0] * self.max_document_length
            arr[:len(x_raw[i])] = x_raw[i]
            x_test.append(arr)
            arr = [0] * self.max_document_char_length
            arr[:len(x_chars[i])] = x_chars[i]
            x_char.append(arr)
        x_test=np.array(x_test)
        x_char=np.array(x_char)

        # Generate batches for one epoch
        x_input = [(x_test[i], x_char[i]) for i in range(len(x_test))]
        batches = data_helpers.batch_iter(list(x_input), 2, 1, shuffle=False)
#         batches = data_helpers.batch_iter(list(x_test), 1, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = self.sess.run(self.predictions, {self.input_x: [x[0] for x in x_test_batch], self.input_x_char: [x[1] for x in x_test_batch], self.dropout_keep_prob: 1.0})
#             batch_predictions = self.sess.run(self.predictions, {self.input_x: x_test_batch, self.dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        # Print accuracy if y_test is defined
        if y_test is not None:
            correct_predictions = float(sum(all_predictions == y_test))
            print("Total number of test examples: {}".format(len(y_test)))
            print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

        # Save the evaluation to a csv
        if save_path:
            predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
            out_path = os.path.join(save_path, "prediction.csv")
            print("Saving evaluation to {0}".format(out_path))
            with open(out_path, 'w') as f:
                csv.writer(f).writerows(predictions_human_readable)
        return int(all_predictions[0]) + 1
    
if __name__=="__main__":
    d = intent_detector()
    import sys
    start_time = time.time()
    pred_label = d.detect(sys.argv[1], label=None, save_path=sys.argv[2])
    print(pred_label)
    elapsed_time = time.time() - start_time
    print(elapsed_time)