import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load_data import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import cPickle
from sklearn.datasets import fetch_mldata
import csv
from sklearn.cross_validation import train_test_split
srng = RandomStreams()
def floatX(X):
	return np.asarray(X, dtype=theano.config.floatX)
def init_weights(shape):
	return theano.shared(floatX(np.random.randn(*shape) * 0.01))
def rectify(X):
	return T.maximum(X, 0.)
def softmax(X):
	e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
	return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
def dropout(X, p=0.):
	global srng
	if p > 0:
		retain_prob = 1 - p
		X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
		X /= retain_prob
	return X
def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
	grads = T.grad(cost=cost, wrt=params)
	updates = []
	for p, g in zip(params, grads):
		acc = theano.shared(p.get_value() * 0.)
		acc_new = rho * acc + (1 - rho) * g ** 2
		gradient_scaling = T.sqrt(acc_new + epsilon)
		g = g / gradient_scaling
		updates.append((acc, acc_new))
		updates.append((p, p - lr * g))
	return updates
def model(X, w, w2, w3, w4, w_o, p_drop_conv, p_drop_hidden):
	l1a = rectify(conv2d(X, w, border_mode='full'))
	l1 = max_pool_2d(l1a, (2, 2))
	l1 = dropout(l1, p_drop_conv)
	
	l2a = rectify(conv2d(l1, w2))
	l2 = max_pool_2d(l2a, (2, 2))
	l2 = dropout(l2, p_drop_conv)
	
	l3a = rectify(conv2d(l2, w3))
	l3b = max_pool_2d(l3a, (2, 2))
	l3 = T.flatten(l3b, outdim=2)
	l3 = dropout(l3, p_drop_conv)
	
	l4 = rectify(T.dot(l3, w4))
	l4 = dropout(l4, p_drop_hidden)

	pyx = softmax(T.dot(l4, w_o))
	return l1, l2, l3, l4, pyx

def read_data_source(path_name = None,target = None):
	with open(path_name,'rU') as files:
		data = csv.reader(files,delimiter = ',')
		X = []
		y = []
		count = 0
		for rows in data:
			if(count == 0):
				count += 1
				continue
			if(target == None):
				X.append(np.array(rows))
			else:
				X.append(np.array(row[target + 1 : ]))
				y.appen(np.array(row[target]))
	print(str(np.shape(X)))
	print(str(len(y)))
	X = floatX(np.vstack(X))
	X = X/255.
	if(target != None):
		return (X,np.array(y,dtype = np.uint8))
	else:
		return X

def load_model():
	model_list = []
	model_reader = file('model_r/obj.save','rb')
	for i in range(2):
		model_list.append(cPickle.load(model_reader))
	model_reader.close()
	return model_list
def test_op():
	data = read_data_source('test.csv')
	#data = fetch_mldata('MNIST original')
	#data_x = floatX(data.data)
	#data_x = data_x/255.
	#y = np.array(data.target,np.uint8)
	#train_X,test_X,train_y,test_y = train_test_split(data_x,y,test_size = .60)
	models = load_model()
	data = data.reshape(-1,1,28,28)
	pred_y = models[1](data)
	#print(np.mean(pred_y == train_y))
	
	with open('prediction/conv_prediction.txt','w+') as prediction_result:
		for each in pred_y:
			prediction_result.write(str(each)+'\n')
	
	return
def main_op():
	print('start loading data')
	trX, teX, trY, teY = mnist(one_hot_T = True)
	trX = trX.reshape(-1, 1, 28, 28)
	teX = teX.reshape(-1, 1, 28, 28)
	X = T.ftensor4()
	Y = T.fmatrix()
	w = init_weights((32, 1, 3, 3))
	w2 = init_weights((64, 32, 3, 3))
	w3 = init_weights((128, 64, 3, 3))
	w4 = init_weights((128 * 3 * 3, 625))
	w_o = init_weights((625, 10))

	noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, w_o, 0.2, 0.5)
	l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, w_o, 0., 0.)
	y_x = T.argmax(py_x, axis=1)
	cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
	params = [w, w2, w3, w4, w_o]
	updates = RMSprop(cost, params, lr=0.001)
	train = theano.function(inputs = [X, Y], outputs = cost, updates = updates, allow_input_downcast = True)
	predict = theano.function(inputs = [X], outputs = y_x, allow_input_downcast = True)
	print('end of symbol formation')
	count = 0
	#file_writer =  open('conv_nnet_file_output.txt','a+')
	for i in range(100):
		file_writer =  open('conv_nnet_file_output.txt','a+')
		try:
			file_writer.write('start '+ str(count) + ' iteration for training: ' )
			for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
				cost = train(trX[start:end], trY[start:end])
			file_writer.write(str(np.mean(np.argmax(teY, axis=1) == predict(teX))) +' ' + str(len(teX)) )
			count += 1
			file_writer.write('end of operation')
		finally:
			file_writer.close()
			with open('model/obj.save','wb') as model_writer:
				cPickle.dump(train,model_writer,protocol = cPickle.HIGHEST_PROTOCOL)
				cPickle.dump(predict,model_writer,protocol = cPickle.HIGHEST_PROTOCOL)
	
	return
#main_op()
test_op()
