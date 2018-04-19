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
	if(tf.is_nan(weight) == True):
		exit()
	else:
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

def findFraud(input1,input2):
	fraud1 = []
	fraud2 = []
	avg1 = (np.sum(input1) / len(input1)) + 10
	avg2 = (np.sum(input2) / len(input2)) + 10
	sum1 = 0.0
	sum2 = 0.0
	for in1,in2 in zip(input1,input2):
		sum1 += (in1 - avg1)**2.0
		sum2 += (in2 - avg2)**2.0
	variance1 = sum1 / len(input1)
	variance2 = sum2 / len(input2)
	deviaton1 = math.sqrt(variance1) * 2.0
	deviaton2 = math.sqrt(variance2) * 1.0
	avg1 += deviaton1
	avg2 += deviaton2
	print '\nFraud Avg:', avg1, '\n'
	print '\nFraud Avg:', avg2, '\n'
	for in1,in2 in zip(input1,input2):
		if((in1 > avg1 or in1 < (avg1*-1.0)) or (in2 > avg2 or in2 < (avg2*-1.0))):
			fraud1.append(in1)
			fraud2.append(in2)
	return zip(fraud1,fraud2)


enough = 0
oldCost = 0
oldWeight = 0
oldBias = 0 
rnd = np.random

readCSV = pd.read_csv('train1.csv',delimiter=',', names=['x_axis','y_axis'])
readCSV = readCSV[1:251]
train_X = readCSV.x_axis
train_Y = readCSV.y_axis
frauds = zip(*findFraud(train_X,train_Y))
train_X = pd.Series(list(set(train_X) - set(frauds[0])))
train_Y = pd.Series(list(set(train_Y) - set(frauds[1])))
n_samples = train_X.shape[0]
new_learning_rate = 0.001 / n_samples
learning_rate = tf.placeholder('float')
training_epochs = 100
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
	plt.plot(frauds[0], frauds[1], 'ro', label='Fraud data')
	plt.plot(train_X, sess.run(w) * train_X + sess.run(b),'b--', label='Fitted line')
	plt.legend()
	plt.show()
