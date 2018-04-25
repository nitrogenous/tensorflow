import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  
def setLearningRate(weight):
	global new_learning_rate
	global n_samples
	# if(tf.is_nan(weight) == True):
	# 	exit()
	# else:
	new_learning_rate = np.abs((0.001 / n_samples) / weight)
	return

def isEnough():
	global enough
	enough += 1
	print '\n',enough,'\n'
	
def oldArgs(cost,weight,bias):
	global oldCost
	global oldWeight
	global oldBias
	oldCost = cost
	oldWeight = weight
	oldBias = bias
	# print  oldCost, oldWeight, oldBias, '\n'
	return;

	variance1 = sum1 / len(input1)
	for in1,in2 in zip(input1,input2):
def delOutlier(input1):
	input1.reset_index(drop=True)
	realElements = [] 
	outliers = []
	avg = (np.sum(input1.x_axis)+np.sum(input1.y_axis)/len(input1.x_axis)*2)
	sums = 0.0
	for i in range(0,len(input1)):
		sums = ((input1.x_axis[i] + input1.y_axis[i]) - avg)**2
	variance1 = sums / (len(input1.x_axis)*2)
	deviaton1 = math.sqrt(variance1)
	avg += deviaton1
	print '\nFraud Avg:', avg, '\n'
	armut = input1
	outlier = []
	for i in range(0,len(input1)):
		if input1.x_axis[i] > avg or input1.x_axis[i] < (avg*-1.0):# or input1.y_axis[i] > avg2 or input1.y_axis[i] < (avg2*-1.0):
			armut = armut.drop(input1.index[i])
			outlier.append(input1.iloc[i])
	outlier = pd.DataFrame(list(outlier)).reset_index(drop=True)
	armut = pd.DataFrame(armut).reset_index(drop=True)
	print outlier
	print armut
	return (armut,outlier)

enough = 0
oldCost = 0
oldWeight = 0
oldBias = 0 
rnd = np.random

readCSV = pd.read_csv('train.csv',delimiter=',', names=['x_axis','y_axis'])
readCSV = readCSV[0:150]
cleanCSV = delOutlier(readCSV)
train = pd.DataFrame(data=cleanCSV[0])
train_X = train.x_axis
train_Y = train.y_axis
outlier = pd.DataFrame(data=cleanCSV[1])
outlier_X = outlier.x_axis
outlier_Y = outlier.y_axis

n_samples = train_X.shape[0]
new_learning_rate = 0.001 / n_samples
learning_rate = tf.placeholder('float')
training_epochs = 50
display_step = 1

X = tf.placeholder('float')
Y = tf.placeholder('float')

w = tf.Variable(np.random.uniform(-float((n_samples/2)),float(n_samples)), name='weight')
b = tf.Variable(0.0, name='bias')

pred = tf.add(tf.multiply(X,w),b)
cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)

init = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run(init)
	print 'Weight', sess.run(w)

	for epoch in range (training_epochs):

		optimizer = tf.train.GradientDescentOptimizer(learning_rate=new_learning_rate).minimize(cost)
		
		for (x,y) in zip(train_X,train_Y):
			sess.run(optimizer, feed_dict={X: train_X, Y: train_Y,learning_rate: new_learning_rate})
			sess.run(cost, feed_dict={X: train_X, Y: train_Y,learning_rate: new_learning_rate})

		if (epoch+1) % display_step == 0 or (epoch+1) <= 10:
			c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
			print('Epoch:', '%07d' % (epoch+1), 'Cost=', '%07f' % c, \
                'Weight=', '%07f' % sess.run(w), 'Bias=', '%07f' % sess.run(b),'Learnin-Rate=', new_learning_rate)

		if('%07f' % sess.run(cost, feed_dict={X: train_X, Y:train_Y}) == oldCost or '%07f' % sess.run(w) == oldWeight or '%07f' % sess.run(b) == oldBias):
			print '\n',sess.run(cost, feed_dict={X: train_X, Y:train_Y}) ,'==', oldCost ,'and', sess.run(w) ,'==', oldWeight ,'and', sess.run(b) ,'==', oldBias
			isEnough()

		if enough >= 3:
			break
		setLearningRate(sess.run(w))
		oldArgs('%07f' % sess.run(cost, feed_dict={X: train_X, Y:train_Y}),'%07f' % sess.run(w),'%07f' % sess.run(b))

	training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
	print "Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n'
	plt.plot(train_X, train_Y, 'go', label='Original data')
	plt.plot(outlier_X, outlier_Y, 'ro', label='Outlier data')
	plt.plot(train_X, sess.run(w) * train_X + sess.run(b),'b--', label='Fitted line')
	plt.legend()
	plt.show()
