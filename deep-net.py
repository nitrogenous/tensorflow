
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib  
matplotlib.use('TkAgg') #for macOs 'python is not a framework error'
import matplotlib.pyplot as plt  

def setLearningRate(weight):
	global new_learning_rate
	global n_samples
	# if(tf.is_nan(weight) == True):
	# 	exit()
	# else:
	new_learning_rate = np.abs((0.0001 / n_samples) / weight)
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

# def delOutlier(input1):
# 	input1.reset_index(drop=True)
# 	realElements = [] 
# 	outliers = []
# 	print input1
# 	avg = (np.sum(input1.x_axis)/len(input1.x_axis))
# 	avg2 = (np.sum(input1.y_axis)/len(input1.y_axis))
# 	print '1. Avg',avg
# 	print '2. Avg',avg2

# 	# print '\n', avg, '\n'

# 	sums = 0.0
# 	sums2 = 0.0
# 	for i in range(0,len(input1)):
# 		sums += math.sqrt(np.abs(input1.x_axis[i] - avg))
# 		sums2 += math.sqrt(np.abs(input1.y_axis[i] - avg))
# 	print '1. Sums', sums
# 	print '2. Sums', sums2
# 	variance1 = (sums / (len(input1))) 
# 	variance2 = (sums2 / (len(input1))) 
# 	avg += variance1
# 	avg2 += variance2
# 	print 'Variance1',variance1
# 	# deviaton1 = (variance1)/2
# 	# print 'deviaton1',deviaton1
# 	# avg += deviaton1
# 	armut = input1
# 	outlier = []
# 	for i in range(0,len(input1)):
# 		if (((input1.x_axis[i] > avg) or (input1.x_axis[i] < (avg*-1.0))) or ((input1.y_axis[i] > avg2) or (input1.y_axis[i] < (avg2*-1.0)))):
# 			armut = armut.drop(input1.index[i])
# 			outlier.append(input1.iloc[i])
# 	outlier = pd.DataFrame(list(outlier)).reset_index(drop=True)
# 	armut = pd.DataFrame(armut).reset_index(drop=True)
# 	print outlier
# 	print armut
# 	return (armut,outlier)

# def delOutlier(Input):
	# Input = Input.sort_values(['x_axis','y_axis'])
	# Input = Input.reset_index(drop=True)
	# realElements = Input
	# outliers = []
	# Q1 = Input.quantile(0.25)
	# Q3 = Input.quantile(0.75)
	# # print Q1,Q3
	# X_AvgM = Q1 - 1.5*(Q3-Q1)
	# X_AvgP = Q3 + 1.5*(Q3-Q1)
	# # Q1 = Input.y_axis.quantile(0.25)
	# # Q3 = Input.y_axis.quantile(0.75)
	# # print Q1,Q3
	# # print realElements
	# # Y_AvgM = Q1 - 1.5*(Q3-Q1)
	# # Y_AvgP = Q3 + 1.5*(Q3-Q1)	
	# print X_AvgM,X_AvgP
	# # print Y_AvgM,Y_AvgP
	# for i in range(0,len(Input)):
	# 	if (Input.x_axis[i] > X_AvgP.x_axis or Input.x_axis[i] < X_AvgP.x_axis) and (Input.y_axis[i] > X_AvgP.y_axis or Input.y_axis[i] < X_AvgM.y_axis):
	# 		print Input.iloc[i]
	# 		outliers.append(Input.iloc[i])
	# 		realElements = realElements.drop(Input.index[i])
	# print '\n',outliers,'\n'
	# print '\n',realElements,'\n'
	# return (realElements,outliers)
def cleanList(hay,needle):
	# print hay
	# hay = hay.sort_values()
	# hay = hay.reset_index(drop=True)
	needles = []
	for i in range(len(hay)):
		if(hay[i] < needle[0] or hay[i] > needle[1]):
			print needle[0], needle[1],hay[i],i
			needles.append(i)
	return needles

def outlierLimits(lst):
	lst = lst.sort_values()
	lst = lst.reset_index(drop=True)
	# print lst.values
	Q1 = lst[math.trunc(len(lst)/4)]
	Q3 = lst[math.trunc((len(lst)/2) + (len(lst)/4))]
	low = Q1 - 4*(Q3-Q1)
	high = Q3 + 4*(Q3-Q1)
	print low,high
	return (low,high)

enough = 0
oldCost = 0
oldWeight = 0
oldBias = 0 
rnd = np.random

readCSV = pd.read_csv('train1.csv',delimiter=',', names=['x_axis','y_axis'])
readCSV = readCSV[0:1000]
readCSV = readCSV.sort_values(['x_axis','y_axis'])
readCSV = readCSV.reset_index(drop=True)
outliers_X = pd.DataFrame(cleanList(readCSV.x_axis,outlierLimits(readCSV.x_axis)))
outliers_Y = pd.DataFrame(cleanList(readCSV.y_axis,outlierLimits(readCSV.y_axis)))
print outliers_X
print outliers_Y
# exit()
outliers = []
train = readCSV

for i in range(len(outliers_X)):
	# print type(train.iloc[outliers_Y[i]])
	outliers.extend([train.iloc[i]])
	train = train.drop(outliers_X.iloc[i])


for i in range(len(outliers_Y)):
	# print type(train.iloc[outliers_Y[i]])
	outliers.extend([train.iloc[i]])
	train = train.drop(outliers_Y.iloc[i])

print len(outliers_Y)
print 'AAA',outliers
outliers = pd.DataFrame(outliers)
print outliers
# outliers = outliers.sort_values(['x_axis','y_axis'])
# outliers = outliers.reset_index(drop=True)
# print train
# print pd.DataFrame(train)
# # exit()

train_X = train.x_axis
train_Y = train.y_axis

n_samples = train_X.shape[0]
new_learning_rate = 0.001 / n_samples
learning_rate = tf.placeholder('float')
training_epochs = 25
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
	plt.plot(outliers.x_axis, outliers.y_axis, 'ro', label='Outlier data') 
	plt.plot(train_X, sess.run(w) * train_X + sess.run(b),'b--', label='Fitted line')
	plt.legend()
	plt.show()
