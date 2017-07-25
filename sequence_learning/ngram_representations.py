"""Class for getting vector representation of ngrams.
Representations are learned from corpus of protein sequnces such that
each sequence is in its own line and is represented as sequence of
ngrams. (data from nonoverlapping_mapping preprocessing)

Representations are one of:
LSA
NMF
word2vec
GloVe
"""

import gensim
import logging
import numpy
import os
import pickle
from gensim import corpora, models, similarities
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from six import iteritems
from time import time
import subprocess

class NgramIterable(object):
	"""Class used for iterating through all ngrams in sequences"""
	def __init__(self, file_name):
		self.file_name = file_name

	def __iter__(self):
		with open(self.file_name) as f:
			for line in f:
				yield line.split()

class SequenceIterable(object):
	def __init__(self, file_name):
		self.file_name = file_name

	def __iter__(self):
		with open(self.file_name) as f:
			for line in f:
				yield line

# Used for doc2vec
class LabeledNgramIterable(object):
	def __init__(self, filename):
		self.filename = filename

	def __iter__(self):
		for uid, line in enumerate(open(self.filename)):
			yield gensim.models.doc2vec.LabeledSentence(words=line.lower().split(), tags=['SENT_%s' % uid])

class NgramRepresentation(object):
	"""Class for getting vector representation of ngram.
	Based on provided model name (one of tfidf, NMF, word2vec,GloVe, doc2vec)
	it translates each ngram to vector of fixed dimension from provided
	protein sequences corpus.
	:params file_ngram_sequences: Path to protein sequnces txt file containing each sequence in
		separate line. Each sequence is represented by sequence of ngrams divided by space.
	:params result_dir: Name of directory to which temp result and finished model will be saved.
	:params extension: Name used in saving temp files and models.
	"""

	def __get_file_name(self, ending, middle = "", no_ending = False):
		result = self.result_dir + '/' + self.extension
		if len(middle):
			result = result + '_' + middle
		if no_ending:
			return result
		return result + '.' + ending


	def	__init__(self, file_ngram_sequences, result_dir, extension):
		self.file_ngram_sequences = file_ngram_sequences
		self.sentences = NgramIterable(file_ngram_sequences)
		#Used for doc2vec
		self.labeled_sentences = LabeledNgramIterable(file_ngram_sequences)
		texts = [[word for word in sentence] for sentence in self.sentences]
		self.result_dir = result_dir
		self.extension = extension
		# dictionary contains unique tokens
		print ("Getting sequence dictionary...")
		if not os.path.exists(self.__get_file_name('dict')):
			t0 = time()
			dictionary = corpora.Dictionary(self.sentences)
			print("done in %0.3fs." % (time() - t0))
			# corpus represents list of sequences where each sequence is represented
			# as list of touples (ngram_id, number_of_occurences)
			corpus = [dictionary.doc2bow(text) for text in texts]
	    	# Save temp results.
			dictionary.save(self.__get_file_name('dict'))
			corpora.MmCorpus.serialize(self.__get_file_name('mm'), corpus)

	def get_corpus(self):
		return corpora.MmCorpus(self.__get_file_name('mm'))

	def get_dictionary(self):
		return corpora.Dictionary.load(self.__get_file_name('dict'))

	@classmethod
	def load_dictionary(self, models_dir, model_prefiks):
		return corpora.Dictionary.load(models_dir + '/' + model_prefiks + '.dict')

	def get_tfidf(self):
		corpus = self.get_corpus()
		if os.path.exists(self.__get_file_name('tfidf')):
			tfidf = models.TfidfModel.load(self.__get_file_name('tfidf'))
		else:
			print ("Getting tfidf model using gensim library...")
			t0 = time()
			dictionary = self.get_dictionary()
			tfidf = models.TfidfModel(corpus)
			print("done in %0.3fs." % (time() - t0))
			tfidf.save(self.__get_file_name('tfidf'))
		return tfidf[corpus]

	def get_ngrams_to_id(self):
		"""Returns map containing ngrams mapped to id"""
		dictionary = self.get_dictionary()
		return dictionary.token2id

	def get_tfidf_of_ngram_in_sequence(self, sequence_id, ngram, tfidf_corpus, ngrams_to_id):
		"""Returns tfidf value of ngram in sequence in corpus.
		:params sequence_id: index of sequence in corpus.
		:params ngram: String name of ngram
		:params tfidf_corpus: From get_tfidf()
		:ngrams_to_id: map containing ngrams to id
		"""
		sequence = tfidf_corpus[sequence_id]
		ngram_id = ngrams_to_id[ngram]
		for id, tfidf in sequence:
			if id == ngram_id:
				return tfidf
		return 0

	def save_sparse_csr(self, filename,array):
		numpy.savez(filename,data = array.data ,indices = array.indices, indptr = array.indptr, shape = array.shape)

	def load_sparse_csr(self, filename):
		loader = numpy.load(filename)
		return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

	def get_tfidf_sparse_matrix(self):
		if not os.path.exists(self.__get_file_name('tfidf_sparse_matrix')):
			self.save_sparse_csr(self.__get_file_name('tfidf_sparse_matrix'), self.get_dictionary_tfidf_representation())
		return self.load_sparse_csr(self.__get_file_name('tfidf_sparse_matrix') + '.npz')

	def get_dictionary_tfidf_representation(self):
		"""Returns matrix as row sparse matrix where rows represent ngrams, and columns sequences.
		(i,j) of matrix is tfidf value of ngram of id = i in sequence of index j.
		"""
		number_of_ngrams = len(self.get_ngrams_to_id())
		sequence_index = 0
		tfidf_corpus = self.get_tfidf()
		# Rows are ngrams, columns are sequences.
		result = numpy.empty([number_of_ngrams, len(tfidf_corpus)])
		col = 0
		for sequence in tfidf_corpus:
			row = 0
			sequence_row = []
			for ngram_touple in sequence:
				# Append 0 if ngram does not appear
				result[row:ngram_touple[0], col] = numpy.zeros(ngram_touple[0] - row)
				result[ngram_touple[0], col] = ngram_touple[1]
				row = ngram_touple[0] + 1
			col = col + 1
		return sparse.csr_matrix(result)

	def get_LSI(self, vector_dim = 100):
		"""Returns array containing ngrams as vectors of dimension vector_dim
		Order of ngrams correspond to ngrams id from get_ngrams_to_id.
		"""
		if os.path.exists(self.__get_file_name('lsi')):
			lsi = models.LsiModel.load(self.__get_file_name('lsi'))
		else:
			print ("Getting LSI model using gensim library...")
			t0 = time()
			corpus_tfidf = self.get_tfidf()
			dictionary = corpora.Dictionary.load(self.__get_file_name('dict'))
			# Additional params chunksize=1, distributed=True
			lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=vector_dim)
			lsi.save(self.__get_file_name('lsi'))
			print("done in %0.3fs." % (time() - t0))
		print ('LSI computed...')
		return lsi.projection.u

	@staticmethod
	def get_ngram_LSI_representation(ngram, ngrams_to_id, u):
		"""Returns vector representation of given ngram.
		:params ngram: String representing ngram
		:params ngrams_to_id: Mao provided from get_ngrams_to_id()
		:params u: Projection from get_LSI()
		"""
		return u[ngrams_to_id[ngram]]


    # Deprecated. Do not use, use efficient version
	def get_NMF(self, vector_dim, max_iter = 50):
		"""Returns matrix W from nmf decomposition."""
		if os.path.exists(self.__get_file_name('pkl', "NMF")):
			w = joblib.load(self.__get_file_name('pkl', "NMF"))
		else:
			print ("Getting NMF model using gensim tfidf matrix...")
			t0 = time()
			nmf = NMF(n_components = vector_dim, init="nndsvd", random_state = 1, max_iter = max_iter)
			w = nmf.fit_transform(self.get_tfidf_sparse_matrix())
			joblib.dump(w, self.__get_file_name('pkl', "NMF"))
			print("done in %0.3fs." % (time() - t0))
		return w

	def get_tfidf_vectorizer(self):
		if os.path.exists(self.__get_file_name('pkl', "tfidf_scikit")) and os.path.exists(self.__get_file_name('pkl', "tfidf_vectorizer_scikit")):
			tfidf = joblib.load(self.__get_file_name('pkl', "tfidf_scikit"))
			tfidf_vectorizer = joblib.load(self.__get_file_name('pkl', "tfidf_vectorizer_scikit"))
		else:
			print("Getting the tfidf using scikit library...")
			t0 = time()
			tfidf_vectorizer = TfidfVectorizer(self.get_ngrams_to_id())
			tfidf = tfidf_vectorizer.fit_transform(SequenceIterable(self.file_ngram_sequences))
			joblib.dump(tfidf, self.__get_file_name('pkl', "tfidf_scikit"))
			joblib.dump(tfidf_vectorizer, self.__get_file_name('pkl', "tfidf_vectorizer_scikit"))
			print("done in %0.3fs." % (time() - t0))
		return tfidf_vectorizer, tfidf

	def get_NMF_efficient(self, vector_dim, max_iter = 50):
		tfidf_vectorizer, tfidf = self.get_tfidf_vectorizer()
		if os.path.exists(self.__get_file_name('pkl', "NMF_efficient")):
			nmf = joblib.load(self.__get_file_name('pkl', "NMF_efficient"))
		else:
			print("Getting the NMF model with tf-idf features...")
			t0 = time()
			nmf = NMF(n_components = vector_dim, init="nndsvd", random_state = 1, max_iter = max_iter).fit(tfidf)
			joblib.dump(nmf, self.__get_file_name('pkl', "NMF_efficient"))
			print("done in %0.3fs." % (time() - t0))
		return tfidf_vectorizer.vocabulary_, numpy.transpose(nmf.components_)

	@staticmethod
	def get_ngram_NMF_representation(ngram, ngrams_to_id, W):
		"""Returns vector representation of given ngram.
		:params ngram: String representing ngram
		:params ngrams_to_id: Map provided from get_ngrams_to_id()
		:params u: Matrix W from get_NMF
		"""
		return W[ngrams_to_id[ngram]]

	# TODO(ana): min_count?
	def get_word2vec(self, vector_dim, window_size, workers = 1):
		"""Creates word2vec model and returns it.
		:params vector_dim: Dimension of vector representation
		:params window_size: Size of context window
		:params workers: Used for training paralelization, only if Cython installed.
		:returns Word2Vec model with fitted sentences.
		"""
		if os.path.exists(self.__get_file_name('word2vec')):
			word2vec = models.Word2Vec.load(self.__get_file_name('word2vec'))
		else:
			print("Getting the word2vec model with tf-idf features...")
			t0 = time()
			word2vec = models.Word2Vec(self.sentences, size=vector_dim, window=window_size, workers = workers, min_count = 1, sg = 1)
			word2vec.save(self.__get_file_name('word2vec'))
			print("done in %0.3fs." % (time() - t0))
		return word2vec

	@staticmethod
	def get_ngram_word2vec_representation(ngram, word2vec):
		"""Returns vector representation of given ngram.
		:params ngram: String representing ngram
		:params word2vec: gensim model from get_word2vec
		"""
		return word2vec[ngram]

	def get_Glove(self, vector_dim, window_size):
		"""Creates txt file containing glove representation of ngrams and returns map ngram to vector.
		:params vector_dim: Dimension of vector representation
		:params window_size: Size of context window
		:returns Map mapping ngram to vector.
		"""
		if not os.path.exists('Glove/results/' + self.extension + '_glove_vectors.txt'):
			print("Getting the word2vec model with tf-idf features...")
			t0 = time()
			process = subprocess.Popen('./Glove/glove.sh %s %s %s %s' % (self.file_ngram_sequences, 'Glove/results/' + self.extension + '_glove_vectors',
				str(vector_dim), str(window_size),), shell=True, stdout=subprocess.PIPE)
			process.wait()
			print("done in %0.3fs." % (time() - t0))

		vectors = {}
		with open('Glove/results/' + self.extension + '_glove_vectors.txt') as f:
			for line in f:
				ngram = line.split()
				vectors[ngram[0]] = numpy.array([float(x) for x in ngram[1:]])
		return vectors

	@staticmethod
	def get_ngram_Glove_representation(ngram, glove):
		"""Returns Glove vector representation of given ngram.
		:params ngram: String representing ngram
		:params glove: Glove model
		"""
		return glove[ngram]

	@classmethod
	def load_model(self, model_name, models_dir, model_prefiks):
		start = models_dir + '/' + model_prefiks
		if model_name == 'LSA':
			return models.LsiModel.load(start + '.lsi').projection.u
		if model_name == 'NMF':
			tfidf = joblib.load(start + '_tfidf_scikit.pkl')
			tfidf_vectorizer = joblib.load(start + '_tfidf_vectorizer_scikit.pkl')
			nmf = joblib.load(start + '_NMF_efficient.pkl')
			return tfidf_vectorizer.vocabulary_, numpy.transpose(nmf.components_)
		if model_name == 'word2vec':
			return models.Word2Vec.load(start + '.word2vec')
		if model_name == 'Glove':
			vectors = {}
			#TODO MAKE THIS MORE GENERAL
			#glove_dir = models_dir[:models_dir.rfind('/')] + '/Glove/results/'
			glove_dir = models_dir + '/Glove/results/'
			with open(glove_dir + model_prefiks + '_glove_vectors.txt') as f:
				for line in f:
					ngram = line.split()
					vectors[ngram[0]] = numpy.array([float(x) for x in ngram[1:]])
			return vectors
