from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import pdb

class dict_adjuster(object):
	def __init__(self, loss_r, loss_c, wordid2r, wordid2c):
		#pdb.set_trace()
		self.loss_r = loss_r
		self.loss_c = loss_c
		self.wordid2r = wordid2r
		self.wordid2c = wordid2c
		self.vocab_size = len(loss_r)
		self.lightrnn_size = int(math.sqrt(self.vocab_size))
		# Make sure that vocab_size is lightrnn_size * lightrnn_size
		assert self.lightrnn_size*self.lightrnn_size == self.vocab_size

	# TODO get random edge from U
	def get_first_edge(self):
		for key in self.U:
			if key < self.vocab_size and self.U[key]:
				return (key, next(iter(self.U[key])))
		return None
	
	def get_loss(self, edge):
		a,b = edge
		return self.loss_r[a][(b-self.vocab_size)//self.lightrnn_size] + self.loss_c[a][(b-self.vocab_size)%self.lightrnn_size] if a < b else self.loss_r[b][(a-self.vocab_size)//self.lightrnn_size] + self.loss_c[b][(a-self.vocab_size)%self.lightrnn_size]
	
	def try_match(self, edge):
		a,b = edge
		#print("try matching %d and %d" %(a, b))
		checked_a = set()
		checked_b = set()
		#pdb.set_trace()
		while a not in self.M_nodes and b not in self.M_nodes and (self.U[a] or self.U[b]):
			#pdb.set_trace()
			c = next(iter(self.U[a]), None)
			if a not in self.M_nodes and c is not None:
				self.U[a].remove(c)
				self.U[c].remove(a)
				checked_a.add(c)
				if self.get_loss((a,c)) < self.get_loss((a,b)):
					self.try_match((a,c))
			
			d =  next(iter(self.U[b]), None)
			if b not in self.M_nodes and d is not None:
				self.U[b].remove(d)
				self.U[d].remove(b)
				checked_b.add(d)
				if self.get_loss((b,d)) < self.get_loss((a,b)):
					self.try_match((b,d))
		
		if a in self.M_nodes and b in self.M_nodes:
			pass
		elif a in self.M_nodes and b not in self.M_nodes:
			for d in checked_b:
				if d not in self.M_nodes:
					self.U[b].add(d)
					self.U[d].add(b)
		elif b in self.M_nodes and a not in self.M_nodes:
			for c in checked_a:
				if c not in self.M_nodes:
					self.U[a].add(c)
					self.U[c].add(a)
		else:
			#pdb.set_trace()
			self.M_nodes.add(a)
			self.M_nodes.add(b)
			if a < b:
				self.M_edges[a] = b
			else:
				self.M_edges[b] = a
	
	def appx_adjust(self):
		
		self.M_edges = {key : None for key in range(self.vocab_size)}
		self.M_nodes = set()
		# U is dict for unmatched edges, key is node A, value is a set of all unmatched nodes that incident on node A
		# In this bipartite graph, 0,...,vocab_size-1, match to vocab_size,...,2*vocab_size-1 
		U_l = {key: set(range(self.vocab_size, 2*self.vocab_size)) for key in range(self.vocab_size)}
		U_r = {key: set(range(self.vocab_size)) for key in range(self.vocab_size, 2*self.vocab_size)}
		self.U = U_l.copy()
		self.U.update(U_r)
	
		e = self.get_first_edge()
		#pdb.set_trace()
		while e:
			print("%d nodes done" % (len(self.M_nodes)/2))
			a,b = e
			self.U[a].remove(b)
			self.U[b].remove(a)
			self.try_match(e)
			e = self.get_first_edge()
		
		total_loss = 0
		total_adjustion = 0
		id2wordid = np.zeros(self.vocab_size)
		wordid2r = np.zeros(self.vocab_size)
		wordid2c = np.zeros(self.vocab_size)
		for i in range(self.vocab_size):
			id2wordid[self.wordid2r[i] * self.lightrnn_size + self.wordid2c[i]] = i
		#pdb.set_trace()
		for e1, e2 in self.M_edges.iteritems():
			value = self.loss_r[e1][(e2-self.vocab_size) // self.lightrnn_size] + self.loss_c[e1][(e2-self.vocab_size) % self.lightrnn_size]
			total_adjustion += int(e1 != e2-self.vocab_size)
			total_loss += value
			wordid2r[id2wordid[e1]] = (e2-self.vocab_size) // self.lightrnn_size
			wordid2c[id2wordid[e1]] = (e2-self.vocab_size) % self.lightrnn_size
		
		return wordid2r, wordid2c, total_loss, total_adjustion
	"""	
	def greedy_adjust(self):
		loss = self.loss
		total_loss = 0
		available_indices = range(self.vocab_size)	
		xy2wordid = {value:key for key,value in self.wordid2xy.items()}
		wordid2xy = {}
		for i in range(self.vocab_size):
			if i % 100 == 0:
				print("%d nodes done" % i)
			min_loss = float("inf")
			min_index = -1
			min_slot = -1
			for index in available_indices:
				tmp_loss = self.loss[index]
				slot = tmp_loss.index(min(tmp_loss))
				if tmp_loss[slot] < min_loss:
					min_loss = tmp_loss[min_slot]
					min_index = index
					min_slot = slot
			total_loss += loss[min_index][min_slot]
			#pdb.set_trace()
			true_id = xy2wordid[(min_index // self.lightrnn_size, min_index % self.lightrnn_size)]
			wordid2xy[true_id] = (min_slot // self.lightrnn_size, min_slot % self.lightrnn_size)
			available_indices.remove(min_index)
			for j in available_indices:
				loss[j][min_slot] = float("inf")
		
		return total_loss
	"""
