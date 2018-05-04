
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib  
matplotlib.use('TkAgg') #for macOs 'python is not a framework error'
import matplotlib.pyplot as plt  
from colorama import Fore, Back, Style

def setLearningRate(weight):
	global new_learning_rate
	global n_samples
	if(tf.is_nan(weight) == True):
		exit()
	else:
		new_learning_rate = np.abs((0.01 / n_samples) / weight)
	return

def isEnough():
	global enough
	enough += 1
	print '\n',Fore.RED,enough,Fore.RESET,'\n'
	
def oldArgs(cost,weight,bias):
	global oldCost
	global oldWeight
	global oldBias
	oldCost = cost
	oldWeight = weight
	oldBias = bias
	# print  oldCost, oldWeight, oldBias, '\n'
	return;

def cleanList(hay,needle):
	# print hay
	# hay = hay.sort_values()
	# hay = hay.reset_index(drop=True)
	hay = pd.DataFrame(hay)
	needle = zip(*needle)
	needles = []
	# print hay
	# print needle
	for i in range(len(hay)):
		if(hay[0][i] < needle[0][0] or hay[0][i] > needle[0][1] or hay[1][i] < needle[1][0] or hay[1][i] > needle[1][1]):
			# print 'X: ',needle[0][0], needle[0][1],hay[0][i],i
			# print 'Y: ',needle[1][0], needle[1][1],hay[1][i],i
			needles.append(i)
	return needles

def outlierLimits(lst):
	lst = lst.sort_values()
	lst = lst.reset_index(drop=True)
	# print lst.values
	Q1 = lst[math.trunc(len(lst)/4)]
	Q3 = lst[math.trunc((len(lst)/2) + (len(lst)/4))]
	low = Q1 - 3*(Q3-Q1)
	high = Q3 + 3*(Q3-Q1)
	# print low,high
	return (low,high)

enough = 0
oldCost = 0
oldWeight = 0
oldBias = 0 
rnd = np.random

readCSV = pd.read_csv('train1.csv',delimiter=',', names=['x_axis','y_axis'])
readCSV = readCSV[1:250]
readCSV = readCSV.sort_values(['x_axis','y_axis'])
readCSV = readCSV.reset_index(drop=True)
outliersIndex = cleanList(zip(readCSV.x_axis,readCSV.y_axis),zip(outlierLimits(readCSV.x_axis),outlierLimits(readCSV.y_axis)))
# print len(outliersIndex)
outliers = []
train = readCSV
# print outliersIndex


for i in range(0,(len(outliersIndex))):
	# print '\n',i,'\n'
	outliers.extend([train.loc[int(outliersIndex[i])]])
	# print outliers
	train = train.drop(outliersIndex[i])

outliers = pd.DataFrame(outliers)
# print outliers


train_X = train.x_axis
train_Y = train.y_axis
n_samples = train_X.shape[0]
new_learning_rate = 0.01 / n_samples
learning_rate = tf.placeholder('float')
training_epochs = 9999999
display_step = 1

X = tf.placeholder('float')
Y = tf.placeholder('float')

w = tf.Variable(np.random.uniform(-float((n_samples/2)),float(n_samples)), name='weight')
b = tf.Variable(0.0, name='bias')

pred = tf.add(tf.multiply(X,w),b)
cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)

init = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=new_learning_rate).minimize(cost)

with tf.Session() as sess:

	sess.run(init)
	print 'Weight', sess.run(w)

	for epoch in range (training_epochs):

		
		for (x,y) in zip(train_X,train_Y):
			sess.run(optimizer, feed_dict={X: train_X, Y: train_Y,learning_rate: new_learning_rate})
			sess.run(cost, feed_dict={X: train_X, Y: train_Y,learning_rate: new_learning_rate})

		if (epoch+1) % display_step == 0 or (epoch+1) <= 10:
			c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
			print Fore.RED,'Epoch:' ,Fore.RESET, '%07d' % (epoch+1),Fore.BLUE ,'Cost:' ,Fore.RESET,'%07f' % c, \
                Fore.GREEN , 'Weight:',Fore.RESET, '%07f' % sess.run(w), Fore.CYAN , 'Bias:',Fore.RESET, '%07f' % sess.run(b),Fore.YELLOW,'Learnin-Rate:',Fore.RESET, new_learning_rate

		if('%07f' % sess.run(cost, feed_dict={X: train_X, Y:train_Y}) == oldCost or '%07f' % sess.run(w) == oldWeight or '%07f' % sess.run(b) == oldBias):
			print '\n',sess.run(cost, feed_dict={X: train_X, Y:train_Y}) ,'==', oldCost ,'and', sess.run(w) ,'==', oldWeight ,'and', sess.run(b) ,'==', oldBias
			isEnough()

		if enough >= 3:
			break
		setLearningRate(sess.run(w))
		oldArgs('%07f' % sess.run(cost, feed_dict={X: train_X, Y:train_Y}),'%07f' % sess.run(w),'%07f' % sess.run(b))

	training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
	print Fore.BLUE,"Training Cost:",Fore.RESET, training_cost,Fore.GREEN, "Training Weight:",Fore.RESET, sess.run(w),Fore.CYAN, "Training Bias:",Fore.RESET, sess.run(b), '\n'
	plt.plot(train_X, train_Y, 'go', label='Original data')
	print sess.run(cost, feed_dict={X: train_X, Y: train_Y,learning_rate: new_learning_rate})
	# plt.plot(outliers.x_axis, outliers.y_axis, 'ro', label='Outlier data') 
	plt.plot(train_X, sess.run(w) * train_X + sess.run(b),'b--', label='Fitted line')
	plt.legend()
	plt.show()
