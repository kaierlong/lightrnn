# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import pdb
import random
import math
import sys
import pickle

import numpy as np
import tensorflow as tf
from lightrnn import LightRNN
from data_util import Reader

flags = tf.app.flags
logging = tf.logging

flags.DEFINE_string("data_path", "./data", "data_path")
flags.DEFINE_string("dataset", "ptb", "The dataset we use for training")
flags.DEFINE_string("model_dir", "./1-512-model", "model_path")
flags.DEFINE_string("model_name", "lightRNN-model", "model_path")

flags.DEFINE_integer('num_layers', 1, 'Number of layers in RNN')
flags.DEFINE_integer('num_steps', 20, 'Number of steps for BPTT')
flags.DEFINE_integer('embedding_size', 200, 'Embedding dimension for one word')
flags.DEFINE_integer('hidden_size', 512, 'Number of hidden nodes for one layer')
flags.DEFINE_integer('vocab_size', 10000, 'Size of vocabulary')
flags.DEFINE_integer('lightrnn_size', 100, 'Size of row and column vector to represent the word')
flags.DEFINE_integer('train_valid_ratio', 9, 'ratio of size of train data and valid data, better be multiple of thread_num')
flags.DEFINE_integer('top_num', 3, 'num of top candidates when calculate accuracy')
flags.DEFINE_float("lr_decay_factor", 0.8, "The decay factor for learning rate")
flags.DEFINE_float("initial_lr", 1.0, "The initial learning rate for training model")
flags.DEFINE_float("lstm_keep_prob", 0.5, "The keep rate for lstm layers")
flags.DEFINE_float("input_keep_prob", 0.8, "The keep rate for input layer")
flags.DEFINE_float("input_rc_ratio", 0.5, "The ratio for ground true input_rc")
flags.DEFINE_float("max_grad_norm", 5.0, "The max norm that clip the gradients")
flags.DEFINE_bool("use_adam", True, "Use AdamOptimizer as training optimizer")

FLAGS = flags.FLAGS

class Option(object):
	def __init__(self, mode):
		self.mode = mode
		self.num_layers = FLAGS.num_layers
		self.embedding_size = FLAGS.embedding_size	
		self.hidden_size = FLAGS.hidden_size
		self.vocab_size = FLAGS.vocab_size
		self.lightrnn_size = FLAGS.lightrnn_size
		self.top_num = FLAGS.top_num
		self.initial_lr = FLAGS.initial_lr
		self.max_grad_norm = FLAGS.max_grad_norm
		self.use_adam = FLAGS.use_adam
		self.batch_size = 1
		self.num_steps = FLAGS.num_steps
		self.lstm_keep_prob = 1.0
		self.input_keep_prob = 1.0
		self.input_rc_ratio = FLAGS.input_rc_ratio

def main(_):
	reader = Reader(FLAGS.data_path, FLAGS.dataset, FLAGS.vocab_size)
	wordid2rc_path = os.path.join(FLAGS.model_dir, "wordid2rc.pkl")
	with open(wordid2rc_path, 'rb') as wordid2rc_file:
		reader.wordid2r = pickle.load(wordid2rc_file) 
		reader.wordid2c = pickle.load(wordid2rc_file)
	for i in range(FLAGS.vocab_size):
		reader.id2wordid[reader.wordid2r[i] * FLAGS.lightrnn_size + reader.wordid2c[i]] = i

	predict_opt = Option("predict")
	model = LightRNN(predict_opt)	
	model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
	model_saver = tf.train.Saver(model_variables)

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
		if ckpt_path:
			model_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))
			print("Read model parameters from %s" % tf.train.latest_checkpoint(FLAGS.model_dir))
		else:
			print("model doesn't exists")
		
		sen_input = raw_input("input: ")
		while sen_input:
			raw_inputs=[]
			sen_input = sen_input.split()
			for word in sen_input:
				if(reader.word2id.has_key(word)):
					raw_inputs.append(reader.word2id[word]) 
				else:
					raw_inputs.append(reader.word2id[reader.vocab.UNK])
			sen_len = min(len(raw_inputs), FLAGS.num_steps)
			if sen_len < FLAGS.num_steps:
				raw_inputs.extend([reader.word2id[reader.vocab.UNK]]*(FLAGS.num_steps-sen_len))
			else:
				raw_inputs = raw_inputs[-FLAGS.num_steps:]
			x_r = np.asarray(map(lambda x : reader.wordid2r[x], raw_inputs)).reshape((FLAGS.num_steps, 1))
			x_c = np.asarray(map(lambda x : reader.wordid2c[x], raw_inputs)).reshape((FLAGS.num_steps, 1))	
			feed_dict = {}
			feed_dict[model.data_r] = x_r
			feed_dict[model.data_c] = x_c
			fetches = model.pred_topK
			value = sess.run(fetches, feed_dict)
			print (value)
			print("top 3 answers are:")
			for i in xrange(len(value[0])):
				key = value[sen_len-1][i]
				val = reader.id2wordid[key]
				print(" %s " % reader.vocab.words[val])
			sen_input = raw_input("input: ")

if __name__ == "__main__":
	tf.app.run()

