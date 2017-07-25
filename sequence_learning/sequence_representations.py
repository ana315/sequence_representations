"""Class for getting vector representation of sequence.
Every sequence is represented as sequence of vectors and it is given as input to
LSMT rnn network to predict random/non random sequence.

Script expects file on following format:
	first row contains information about number of sequences and
	max_sequence_length as:
		number_of_sequences=x max_sequence_length=y
	after that follow sequences: every sequence is represented as sequence of
	ngrams divided by space.

Labels can be given either as separate file, such that order f labels correspond
to sequences in file, or at the end of each sequence

After training, sequence representation is extracted from hidden LSTM layer
as vector of size LSTM_dimension.

Sequeces are saved to file:
first line contains information about LSTM length and number of addtional
features of every sequence.
After that each sequence vector is written in new line.
"""
import argparse
import datetime
import multiprocessing
import numpy
import os
import sys

numpy.random.seed(1337)
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Masking
from keras.layers import Merge
from keras.layers import add
from keras.optimizers import RMSprop
from keras.utils import np_utils
from ngram_representations import NgramRepresentation
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('input_file_name',
					help = 'Reprenesents name of file containing ngram sequences.',
					type = str)
parser.add_argument('ngram_models_directory',
					help = """Reprenesents name of directory containing models for
					extracting ngram representations.""",
					type = str)
parser.add_argument('ngram_models_name',
					help = """Reprenesents name of all saved models saved from
					ngram_representations. It assumes that all models have same
					prefixes given with that string: model_names.sthelse""",
					type = str)
parser.add_argument('ngram_vector_dimension',
					help = 'Represents dimension of ngram vector representation',
					type = int)
parser.add_argument('LSTM_dimension',
					help = 'Represents dimension of sequence vector representation',
					type = int)
parser.add_argument('--input_labels_file_name',
					help = """Reprenesents name of file containing labels
					corresponding to sequences in input_file_name. If not specified,
					it expects input_file_name to contain labels
					at the end of each line""",
					type = str)
parser.add_argument('--validation_split',
					help = """Represents float or int: If float, should be between
					0.0 and 1.0 and represent the proportion of the dataset to include
					in the test split. If int, represents the absolute
					number of test samples.""",
					type = float,
					default = 0.2)
parser.add_argument('--append_to_beginning',
					help = 'When extended, data is appended at the beginning',
					)
parser.add_argument('--log_to_file',
					help = 'Whether to print to file instead of console')
#Flags for RNN network
parser.add_argument('--nb_epoch',
					help = 'Number of epochs for LSTM network',
					type = int,
					default = 30)
parser.add_argument('--dropout_out',
					help = 'dropout of output LSTM network',
					type = float,
					default = 0.3)
parser.add_argument('--dropout_in',
					help = 'dropout of input LSTM network',
					type = float,
					default = 0.0)
parser.add_argument('--batch_size',
					help = 'Batch size for LSTM network',
					type = int,
					default = 128)
parser.add_argument('--learning_rate',
					help = 'Learning rate for LSTM network',
					type = float,
					default = 0.0001)
parser.add_argument('--activation',
					help = 'Activation layer between LSTM and Dense layer',
					type = str,
					default = 'relu',
					choices = ['softmax', 'relu', 'softplus', 'softsign', 'relu',
					 		   'tanh', 'sigmoid', 'hard_sigmoid'])
parser.add_argument('--inner_activation',
					help = 'Inner activation layer in LSTM network',
					type = str,
					default = 'hard_sigmoid',
					choices = ['softmax', 'relu', 'softplus', 'softsign', 'relu',
							   'tanh', 'sigmoid', 'hard_sigmoid'])

# All ngram models included
parser.add_argument('--word2vec',
					help = 'Word2vec ngram representation is included')
parser.add_argument('--Glove',
					help = 'Glove ngram representation is included')
parser.add_argument('--LSA',
					help = 'LSA ngram representation is included')
parser.add_argument('--NMF',
					help = 'NMF ngram representation is included')
seed = 6

def get_ngram_representations():
	ngram_rep = []
	if args.word2vec:
		ngram_rep.append('word2vec')
	if args.Glove:
		ngram_rep.append('Glove')
	if args.LSA:
		ngram_rep.append('LSA')
	if args.NMF:
		ngram_rep.append('NMF')
	return ngram_rep

def string_parameters():
	result = str(args.ngram_vector_dimension)
	result += '_LSTM' + str(args.LSTM_dimension)
	result += '_' + args.activation + '_' + args.inner_activation
	result += '_' + str(args.learning_rate) + '_' + str(args.batch_size)
	result += '_' + str(args.nb_epoch)
	result += '_' + str(args.dropout_in)
	result += '_' + str(args.dropout_out)
	return result

class LabeledSequenceIterable(object):
	"""Class used for reading from labeled sequences
	If only one file is given, it expects labels are in the same file at the
	end of each row.
	Otherwise it expects labels are in separate file in the order coresponding to
	order of sequences.
	Input_file_name file should be in the format explained at the beginning.
	"""
	def __init__(self, input_file_name, input_labels_file_name = None):
		self.file_name = input_file_name
		labels = []
		with open(self.file_name) as f:
			first_line = f.readline().split()
			self.number_of_sequences = int(first_line[0].split('=')[-1])
			self.max_sequence_length = int(first_line[1].split('=')[-1])
			if input_labels_file_name is None:
				self.remove_label = True
				for line in f:
					line = line.strip()
					if len(line) == 0:
						continue
					labels.append(int(line[-1]))
			else:
				self.remove_label = False
				with open(input_labels_file_name) as l:
					for line in l:
						line = line.strip()
						if len(line) == 0:
							continue
						labels.append(int(line))
		self.labels = numpy.array(labels)

	def __iter__(self):
		first = False
		with open(self.file_name) as f:
			f.readline()
			for line in f:
				line = line.strip()
				if not len(line): continue
				if self.remove_label:
					yield line[:-2].split()
				else:
					yield line.split()

	def get_labels(self):
		return self.labels

	def get_max_sequence_length(self):
		return self.max_sequence_length

	def get_number_of_sequences(self):
		return self.number_of_sequences

model_names = ['word2vec', 'Glove', 'LSA', 'NMF']
ngram_representations = {
		'LSA':NgramRepresentation.get_ngram_LSI_representation,
		'NMF':NgramRepresentation.get_ngram_NMF_representation,
		'word2vec':NgramRepresentation.get_ngram_word2vec_representation,
		'Glove':NgramRepresentation.get_ngram_Glove_representation
		}

def generate_model_parameters(models_directory, model_prefiks, models = model_names):
	"""Method for generating parameters to be called from get_sequence_x_representation
	Example: models_directory: temp_results, model_prefiks: vecdim10_ctx25
	:param modeles_directory: Name of directory where models are saved.
	:param model_prefiks: All models start with same prefiks.
	:param models: List of models from model_names to include
	:return Dictionary mapping model_name to tuple of parameters
	"""
	model_parameters = {}
	dictionary = NgramRepresentation.load_dictionary(models_directory, model_prefiks)
	ngrams_to_id = dictionary.token2id
	if 'NMF' in models:
		model_parameters['NMF'] = (NgramRepresentation.load_model(
									'NMF', models_directory, model_prefiks), )
	if 'LSA' in models:
		model_parameters['LSA'] = (ngrams_to_id, NgramRepresentation.load_model(
									'LSA', models_directory, model_prefiks))
	if 'word2vec' in models:
		model_parameters['word2vec'] = (NgramRepresentation.load_model(
										'word2vec', models_directory, model_prefiks), )
	if 'Glove' in models:
		model_parameters['Glove'] = (NgramRepresentation.load_model(
										'Glove', models_directory, model_prefiks), )

	return model_parameters

def load_dataset(file_name, vector_dimension, model_parameters,
				label_file_name = None, validation_split = 0.2,
				append_to_beginning = False):
	"""Method used for generating data for LSTM network.
	For each model in models (from model_names) creates separate dataset
	:param file_name: Name of file containing ngram sequences. If label_file_name
			is None it expects this file contains labels 0/1 at end of each line.
	:param vector_dimension: Dimension of vector representation of ngram.
	:param model_parameters: Tuple containing (model_name, parameters of model from
								generate model parameters)
	:param append_to_beginning: If true, 0s are appended to the beginning, otherwise
								to the end
	:param label_file_name: Name of file containing labels corresponding to sequences
	 						in file_name
	:param validation_split: float or int: If float, should be between 0.0 and 1.0 and
	 						represent the proportion of the dataset to include in the
							test split. If int, represents the absolute number of test
							samples.
	:return (X_train, Y_train) tuple if validation_split = 0, (X_train, X_validation,
	 		Y_train, Y_validation) otherwise: Keras compatible dataset.
	"""
	print "\nDataset loading started... \n"
	sequences = LabeledSequenceIterable(file_name, label_file_name)
	limit = sequences.get_number_of_sequences()
	max_sequence_length = sequences.get_max_sequence_length()
	#Y = np_utils.to_categorical(sequences.get_labels())
	Y = sequences.get_labels()
	t0 = time()
	X = numpy.zeros(shape=(limit, max_sequence_length, vector_dimension))
	model_name = model_parameters[0]
	for counter, sequence in enumerate(sequences):
		zero_array = numpy.zeros(
					shape = (max_sequence_length - len(sequence), vector_dimension))
		ngram_rep = [ngram_representations[model_name](ngram, *model_parameters[1])
					for ngram in sequence]
		ngram_rep = numpy.array(ngram_rep)
		#print(ngram_rep)
		if append_to_beginning:
			X[counter] = numpy.concatenate((zero_array, ngram_rep), axis = 0)
		else:
			X[counter] = numpy.concatenate((ngram_rep, zero_array), axis = 0)
	print("done in %0.3fs." % (time() - t0))
	if validation_split > 0:
		sss = StratifiedShuffleSplit(test_size=validation_split, random_state=seed,
									n_splits=1)
		result = sss.split(X, Y)
		for indices in result:
			train_indices, test_indices = indices
		return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]
	return X, Y

def create_bidir_LSTM_model(input_shape, LSTM_dimension,
							activation, inner_activation, learning_rate):
	"""Method used for creating bidirectional one-layer LSTM network.
	:param input_shape: Tuple containing information about input dimension:
						(max_sequence_length, vector_dimension)
	:param LSTM_dimension: Dimension of hidden layer and final dimension on sequence
						   representation.
	:return Keras Sequential model containing 2 LSTM networks merged by Merge layer
			and connected with one Dense layer
	"""
	left = Sequential()
	left.add(Masking(mask_value=0., input_shape=input_shape))
	left.add(LSTM(LSTM_dimension, activation=activation,
			recurrent_activation=inner_activation,
			implementation=1,
			dropout = args.dropout_in))

	right = Sequential()
	right.add(Masking(mask_value=0., input_shape=input_shape))
	right.add(LSTM(LSTM_dimension, activation=activation,
				 recurrent_activation=inner_activation,
				 implementation=1,
				 go_backwards=True,
				 dropout = args.dropout_in))

	model = Sequential()
	model.add(Merge([left, right], mode = 'sum'))
	#model.add(add([left, right]))
	model.add(Dropout(args.dropout_out))
	model.add(Dense(1, activation='sigmoid'))
	rms_opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08)
	model.compile(optimizer=rms_opt,
				  loss='binary_crossentropy',
				  metrics=["accuracy"],)
	return model

def get_input_shape(model):
	"""Returns input shape of bidir model"""
	return model.layers[0].layers[0].layers[1].input_shape[1:]

def train_bidir_LSTM(x_train, y_train, model,
					nb_epoch, batch_size,
					model_file_name, ngram_model_name,
					x_validation = None, y_validation = None):

	"""Method for training bidirectional LSTM model with x_train and y_train data.
	If validation data is provided it uses it for validation.
	:param x_train, y_train: Numpy array containing data from load_dataset
	:param model: Keras bidirectional model from create_bidir_LSTM_model
	:param x_validation, y_validation: Numpy arrays froma load_dataset. If None,
									   validation is not used
	:params nb_epoch, batch_size, learning_rate, train_batch: Params specific for
					Keras models
	"""
	current_epoch = 1
	while current_epoch <= nb_epoch:
		print('\n\nEpoch: {}\n'.format(current_epoch))
		if x_validation is not None:
			model.fit([x_train, x_train],
				y_train,
				validation_data=([x_validation, x_validation], y_validation),
				epochs=1,
				batch_size=batch_size,
				verbose=1)
		else:
			model.fit([x_train, x_train],
				y_train, epochs=1, batch_size=batch_size, verbose=1)
		if current_epoch % 10 == 0:
		  save_model(ngram_model_name, model,
		  			model_file_name, current_epoch,
		   			current_epoch <= 10)
		current_epoch += 1


def get_model_name(ngram_model_name, file_name, ending):
	return 'sequence_results/sequence_models/' + ngram_model_name + '/' + file_name + '.' + ending

def save_model(ngram_model_name, model, file_name, epoch, save_json = False):
	if not os.path.exists('sequence_results/sequence_models/' + ngram_model_name + '/'):
		os.makedirs('sequence_results/sequence_models/' + ngram_model_name + '/')
	if save_json:
		model_json = model.to_json()
		with open(get_model_name(
					ngram_model_name, file_name, 'json'), 'w') as json_file:
		    json_file.write(model_json)
	# serialize weights to HDF5
	file_name = file_name + '__epoch{}'.format(epoch)
	model.save_weights(get_model_name(ngram_model_name, file_name, 'h5'))

def load_from_json_file(model_name, weights_name):
	model_json = open(model_name, 'r').read()
	loaded_model = model_from_json(model_json)
	# load weights into new model
	loaded_model.load_weights(weights_name)
	return loaded_model

def load_model(ngram_model_name, file_name):
	return load_from_json_file(
				get_model_name(ngram_model_name, file_name, 'json'),
				get_model_name(ngram_model_name, file_name, 'h5'))

def LSTM_job(ngram_model_name, model_parameters, validation_split = 0.3):
	"""Method used for getting and training LSTM model based on ngram representations.
	It is used for parallel computing: for each ngram representation
	(LSA, NMF, word2vec, Glove) special core is used for execution.
	:param model_name: String representing name of model ngram representation: one of
						LSA, NMF, word2vec, Glove
	:param model_parameters: Tuple containing parameters need for
							NgramRepresentation.get_ngram_x_representation method.
	:param validation_split: float or int: If float, should be between 0.0 and 1.0
	 						and represent the proportion of the dataset to include in
							the test split. If int, represents the absolute number of
							test samples.
	:return trained model
	"""
	print 'Training LSTM for ' + model_name + " ngram representation...\n"
	x_validation = None
	y_validation = None
	append_to_beginning = False
	if args.append_to_beginning:
		append_to_beginning = True
	if validation_split > 0:
		x_train, x_validation, y_train, y_validation = load_dataset(
									args.input_file_name,
									args.ngram_vector_dimension,
									(model_name, model_parameters),
									args.input_labels_file_name,
								    validation_split,
									append_to_beginning = append_to_beginning)
		c0 = 0
		c1 = 0
		for i, y in enumerate(y_validation):
			if y == 0:
				c0 += 1
			else:
				c1 += 1
		print('\nNumber of positive labels in validation set: ' + str(c1) + '\n')
		print('Number of negative labels in validation set: ' + str(c0) + '\n')
	else:
		x_train, y_train = load_dataset(
									args.input_file_name,
									args.ngram_vector_dimension,
									(model_name, model_parameters),
									args.input_labels_file_name,
									append_to_beginning = append_to_beginning)

	input_shape = (x_train.shape[1], args.ngram_vector_dimension)
	model_file_name = ngram_model_name + string_parameters()
	model = create_bidir_LSTM_model(input_shape, args.LSTM_dimension,
	args.activation, args.inner_activation, args.learning_rate)
	print 'Training LSTM for ' + model_name + ' started:\n\n'
	t0 = time()
	train_bidir_LSTM(x_train, y_train, model,
					 args.nb_epoch, args.batch_size,
					 model_file_name, ngram_model_name,
					 x_validation = x_validation, y_validation = y_validation)
	print 'Training LSTM for ' + model_name + ' finished after ' + str(time() - t0) + "sec"
	return model

def build_model_from_model(model):
	#Left branch: Merge, left, LSTM
	left = model.layers[0].layers[0].layers[1]
	#Right branch:
	right = model.layers[0].layers[1].layers[1]
	weights = []

	for l1, l2 in zip(left.get_weights(), right.get_weights()):
		weights.append(l1 + l2)

	LSTM_dimension = weights[1].shape[0]
	model2 = Sequential()
	model2.add(Masking(mask_value=0., input_shape=left.input_shape[1:]))
	model2.add(LSTM(LSTM_dimension, activation=left.activation.__name__,
	 			weights=weights))
	return model2

def get_sequence_representation(new_model, x, append_to_beginning = False):
	"""Method for getting vector representation of given sequence.
	This is done by building a new model with the activations of the old model.
	This model is truncated after the first layer
	:param new_model: Keras bidirectional LSTM model from build_model_from_model
	:param x: Numpy array representing sequence input
	:return vector representation of sequence x
	"""
	shape = new_model.layers[0].input_shape[1:]
	#TODO!
	if shape[0] - x.shape[0] < 0:
		return numpy.zeros(shape[1])
	zero_array = numpy.zeros(shape = (shape[0] - x.shape[0], shape[1]))
	if append_to_beginning:
		x_extended = numpy.concatenate((zero_array, x), axis = 0)
	else:
		x_extended = numpy.concatenate((x, zero_array), axis = 0)
	X = numpy.zeros(shape=(1, shape[0], shape[1]))
	X[0] = x_extended
	return new_model.predict(X)[0]

def get_representations_file_name(ngram_model_name, file_name, ending):
	return 'sequence_results/sequence_representations/' + ngram_model_name + '/' + file_name + '.' + ending

#OLD DO NOT USE
def save_sequence_representations(model, X, ngram_model_name):
	"""Method for saving sequence representations in file.
	:param model: Keras bidirectional LSTM model from create_bidir_LSTM_model
	:param X: all_data representing all sequences from loaded data set. It is returned
	from method LSTM_job as all_data
	:param model_name: String representing model name for ngram representation: one of
	  NMF, LSA, word2vec, Glove.
	"""
	if not os.path.exists('sequence_results/sequence_representations/'):
		os.makedirs('sequence_results/sequence_representations/')

	file_name = get_representations_file_name(
				ngram_model_name, model_name + string_parameters(), 'data')
	with open(file_name, 'w') as f:
		f.write('dimension=' + str(args.LSTM_dimension) + '\n')
		for rep in get_sequence_representation(ngram_model_name, model, X):
			f.write(' '.join(str(val) for val in rep))
			f.write('\n')
	print 'File saved as ' +  file_name + '\n'

if __name__ == "__main__":
	args = parser.parse_args()

	ngram_reps = get_ngram_representations()
	model_parameters = generate_model_parameters(args.ngram_models_directory,
											 args.ngram_models_name,
											 ngram_reps)
	tasks = [(model_name, model_parameters[model_name]) for model_name in model_parameters]
	for task in tasks:
		if args.log_to_file:
			old_stdout = sys.stdout
			print ('\n\nlogging to file: ')
			log_file_name = 'sequence_results/log_files/' + task[0] + '/' + datetime.datetime.now().strftime("%I_%M%p_%B_%d")
			print(log_file_name + '\n')
			log_file = open(log_file_name, "w")
		try:
			if args.log_to_file:
				sys.stdout = log_file
			print('------------------------------------------------------\n')
			print ('Data Info...\n')
			print('Loading dataset with ngram sequences ' + args.input_file_name)
			if args.input_labels_file_name:
				print('Loading labels from ' + args.input_labels_file_name)
			print('Using ' + task[0] + ' representation of ngrams...')
			print('Loading models: ' + args.ngram_models_directory + '/' + args.ngram_models_name + '*')
			print('Ngram dimension:  ' + str(args.ngram_vector_dimension))
			if args.validation_split:
				print ('For validation split used: ' + str(args.validation_split))
			print('------------------------------------------------------\n')
			print ('RNN Info...\n')
			print('LSTM dimension: ' + str(args.LSTM_dimension))
			print('batch size: ' + str(args.batch_size))
			print('number of epochs: ' + str(args.nb_epoch))
			print('learning rate: ' + str(args.learning_rate))
			print('activation function: ' + args.activation)
			print('inner-activation function: ' + args.inner_activation)
			print('droput in: ' + str(args.dropout_in))
			print('droput out: ' + str(args.dropout_out))
			print('------------------------------------------------------\n\n')
			m = LSTM_job(*task)
		finally:
			if args.log_to_file:
				sys.stdout = old_stdout
				log_file.close()

	#Do Parallel work
	"""pool = multiprocessing.Pool(1)
	results = [pool.apply_async(LSTM_job, t) for t in tasks]

	for result in results:
		name, message = result.get()
		print name + message"""
