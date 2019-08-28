# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pdb
import sys
import pickle
import tensorflow as tf
import numpy as np
from data_util import Reader

flags = tf.app.flags
logging = tf.logging

flags.DEFINE_string("data_path", "./data", "data_path")
flags.DEFINE_string("dataset", "ptb", "The dataset we use for training")
flags.DEFINE_string("model_dir", "./model", "model_path")
flags.DEFINE_integer('vocab_size', 10000, 'Size of vocabulary')
flags.DEFINE_integer('lightrnn_size', 100, 'Size of lightrnn')
flags.DEFINE_integer('batch_size', 256, 'Number of lines in one batch for training')
flags.DEFINE_integer('num_steps', 20, 'Number of steps for BPTT')

FLAGS = flags.FLAGS

reader = Reader(FLAGS.data_path, FLAGS.dataset, FLAGS.vocab_size, FLAGS.batch_size, FLAGS.num_steps)

wordid2rc_path = os.path.join(FLAGS.model_dir, "wordid2rc.pkl")
if os.path.isfile(wordid2rc_path):
	with open(wordid2rc_path, 'rb') as wordid2rc_file:
		reader.wordid2r = pickle.load(wordid2rc_file) 
		reader.wordid2c = pickle.load(wordid2rc_file)

pdb.set_trace()
