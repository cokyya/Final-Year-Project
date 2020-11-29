import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import tensorflow as tf
#from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.python.framework import ops
ops.reset_default_graph()
import pandas as pd
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer


# Create graph
sess = tf.Session()

#load the dataset
with open("trainingData.csv","r") as readfile:
    reader1= csv.reader(readfile)
    read=[]
    for row in reader1:
        if len(row)!=0:
            read=read+[row]

readfile.close
df=pd.DataFrame(read)
#print(df)


#split the dataset
xx_vals=df.iloc[1:,0:520].values
ll_label=df.iloc[1:,529].values
x_vals= xx_vals.astype(np.float)
label=ll_label.astype(np.float)
x_vals[x_vals == 100] = -110
#standardize the scaler
from scipy import stats
x_vals=stats.zscore(x_vals)
print(x_vals)
print(x_vals.shape)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
x_vals_reduced=imp.fit_transform(x_vals)

reduction=520

#To remove dulplicate
#To shuffle the data for K-fold and batch implementation
label = label.reshape((19937, 1))
new_concatenate=np.concatenate((x_vals_reduced, label), axis=1)
uniques_array=np.unique(new_concatenate, axis = 0)
random.shuffle(uniques_array)
np.savetxt("uniques_array.csv", uniques_array, delimiter=",")

#load the dataset
with open("uniques_array.csv","r") as readfile:
    reader1= csv.reader(readfile)
    read=[]
    for row in reader1:
        if len(row)!=0:
            read=read+[row]

readfile.close
df=pd.DataFrame(read)
#print(df)


#split the dataset
xx_vals=df.iloc[:,0:reduction].values
ll_label=df.iloc[:,-1].values
x_vals_reduced= xx_vals.astype(np.float)
label=ll_label.astype(np.float)
print(x_vals_reduced.shape)
print(label.shape)


#construct label matrix 
y_vals1 = np.array([1 if y == 0 else -1 for y in label])
y_vals2 = np.array([1 if y == 1 else -1 for y in label])
y_vals3 = np.array([1 if y == 2 else -1 for y in label])
y_vals4 = np.array([1 if y == 3 else -1 for y in label])
y_vals5 = np.array([1 if y == 10 else -1 for y in label])
y_vals6 = np.array([1 if y == 11 else -1 for y in label])
y_vals7 = np.array([1 if y == 12 else -1 for y in label])
y_vals8 = np.array([1 if y == 13 else -1 for y in label])
y_vals9 = np.array([1 if y == 20 else -1 for y in label])
y_vals10 = np.array([1 if y == 21 else -1 for y in label])
y_vals11 = np.array([1 if y == 22 else -1 for y in label])
y_vals12 = np.array([1 if y == 23 else -1 for y in label])
y_vals13 = np.array([1 if y == 24 else -1 for y in label])
y_vals = np.array([y_vals1, y_vals2, y_vals3,y_vals4,y_vals5,y_vals6,y_vals7,y_vals8,y_vals9,y_vals10,y_vals11,y_vals12,y_vals13])

class1_x = [x[0] for i, x in enumerate(x_vals_reduced)if label[i] == 0]
class1_y = [x[1] for i, x in enumerate(x_vals_reduced) if label[i] == 0]
class2_x = [x[0] for i, x in enumerate(x_vals_reduced) if label[i] == 1]
class2_y = [x[1] for i, x in enumerate(x_vals_reduced) if label[i] == 1]
class3_x = [x[0] for i, x in enumerate(x_vals_reduced) if label[i] == 2]
class3_y = [x[1] for i, x in enumerate(x_vals_reduced) if label[i] == 2]
class4_x = [x[0] for i, x in enumerate(x_vals_reduced) if label[i] == 3]
class4_y = [x[1] for i, x in enumerate(x_vals_reduced) if label[i] == 3]
class5_x = [x[0] for i, x in enumerate(x_vals_reduced)if label[i] == 10]
class5_y = [x[1] for i, x in enumerate(x_vals_reduced) if label[i] == 10]
class6_x = [x[0] for i, x in enumerate(x_vals_reduced) if label[i] == 11]
class6_y = [x[1] for i, x in enumerate(x_vals_reduced) if label[i] == 11]
class7_x = [x[0] for i, x in enumerate(x_vals_reduced) if label[i] == 12]
class7_y = [x[1] for i, x in enumerate(x_vals_reduced) if label[i] == 12]
class8_x = [x[0] for i, x in enumerate(x_vals_reduced) if label[i] == 13]
class8_y = [x[1] for i, x in enumerate(x_vals_reduced) if label[i] == 13]
class9_x = [x[0] for i, x in enumerate(x_vals_reduced)if label[i] == 20]
class9_y = [x[1] for i, x in enumerate(x_vals_reduced) if label[i] == 20]
class10_x = [x[0] for i, x in enumerate(x_vals_reduced) if label[i] == 21]
class10_y = [x[1] for i, x in enumerate(x_vals_reduced) if label[i] == 21]
class11_x = [x[0] for i, x in enumerate(x_vals_reduced) if label[i] == 22]
class11_y = [x[1] for i, x in enumerate(x_vals_reduced) if label[i] == 22]
class12_x = [x[0] for i, x in enumerate(x_vals_reduced) if label[i] == 23]
class12_y = [x[1] for i, x in enumerate(x_vals_reduced) if label[i] == 23]
class13_x = [x[0] for i, x in enumerate(x_vals_reduced)if label[i] == 24]
class13_y = [x[1] for i, x in enumerate(x_vals_reduced) if label[i] == 24]

# Declare batch size
batch_size = 2000

#using Kfold to split
#select 18000 data from 19937
index_select=np.random.choice(len(x_vals_reduced),18000, replace=False)
print(index_select)
x_instance_select=x_vals_reduced[index_select]
y_instance_select=y_vals[:,index_select]
label[label == 0] = 0
label[label == 1] = 1
label[label == 2] = 2
label[label == 3] = 3
label[label == 10] = 4
label[label == 11] = 5
label[label == 12] = 6
label[label == 13] = 7
label[label == 20] = 8
label[label == 21] = 9
label[label == 22] = 10
label[label == 23] = 11
label[label == 24] = 12
print(label)
label_new=label[index_select]
#np.savetxt("x.csv", x_instance_select, delimiter=",")
#np.savetxt("y.csv", y_instance_select, delimiter=",")
#print(x_instance_select.shape)
#print(y_instance_select.shape)
kf = KFold(n_splits=9)

overall_train_accuracy =[]
overall_test_accuracy =[]


for train_index, test_index in kf.split(x_instance_select):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_vals_train = x_instance_select[train_index]
        x_vals_test = x_instance_select[test_index]
        y_vals_train = y_instance_select[:,train_index]
        y_vals_test = y_instance_select[:,test_index]
        label_train=label_new[train_index]
        label_test=label_new[test_index]
        #print(y_vals_train.shape)
        #print(y_vals_test.shape)


        # Initialize placeholders
        x_data = tf.placeholder(shape=[None, reduction], dtype=tf.float32)
        y_target = tf.placeholder(shape=[13, None], dtype=tf.float32)
        prediction_grid = tf.placeholder(shape=[None, reduction], dtype=tf.float32)

        # Create variables for svm

        b = tf.Variable(tf.random_normal(shape=[13, batch_size]))

        # Gaussian (RBF) kernel
        gamma = tf.constant(-32.0)
        dist = tf.reduce_sum(tf.square(x_data), 1)
        dist = tf.reshape(dist, [-1, 1])
        sq_dists = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
        my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))


        # Declare function to do reshape/batch multiplication
        def reshape_matmul(mat, _size):
            v1 = tf.expand_dims(mat, 1)
            v2 = tf.reshape(v1, [13, _size, 1])
            return tf.matmul(v2, v1)

        # Compute SVM Model
        first_term = tf.reduce_sum(b)
        b_vec_cross = tf.matmul(tf.transpose(b), b)
        y_target_cross = reshape_matmul(y_target, batch_size)

        second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)), [1, 2])
        loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))
        #print(first_term.get_shape())

        # Gaussian (RBF) prediction kernel
        rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
        rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
        pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
        pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

        prediction_output = tf.matmul(tf.multiply(y_target, b), pred_kernel)
        prediction = tf.argmax(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target, 0)), tf.float32))

        # Declare optimizer
        my_opt = tf.train.GradientDescentOptimizer(0.01)
        train_step = my_opt.minimize(loss)

        # Initialize variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training loop
        loss_vec = []
        batch_accuracy = []
        test_accuracy =[]
        test_loss_vec = []
        for i in range(100):
            rand_index = np.random.choice(train_index, size=batch_size,replace=False)
            rand_x = x_instance_select[rand_index]
            rand_y = y_instance_select[:, rand_index]
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
            
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            loss_vec.append(temp_loss)
       
            acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
                                                    y_target: rand_y,
                                                    prediction_grid: rand_x})
            batch_accuracy.append(acc_temp)

            
            if (i + 1) % 25 == 0:
               print('Step #' + str(i+1))
               print('Loss = ' + str(temp_loss))

               print(np.mean(batch_accuracy))
            

        print('finish training model')


        #putting all training set into the final SVM model
        train_accuracy =[]
        train_loss_vec = []
        start_t=0
        a = train_index
        #a = np.asarray(a, dtype=np.int32)
        #random.shuffle(a)

        for k in range(8):
            print(k)
            train_batch_index = a[start_t:2000*(k+1)]
            #print(train_batch_index)
            #train_batch_index = np.asarray(train_batch_index, dtype=np.int32)
            #random.shuffle(train_batch_index)
            #print(train_batch_index)
            start_t=2000*(k+1)
            train_batch_x = x_instance_select[train_batch_index]
            #np.savetxt("train_batch_x.csv", x_vals_train, delimiter=",")
            train_batch_y = y_instance_select[:, train_batch_index]
            train_batch_label=label_new[train_batch_index]
            '''
            train_acc_temp = sess.run(accuracy, feed_dict={x_data: train_batch_x,
                                                    y_target: train_batch_y,
                                                    prediction_grid: train_batch_x})
            train_accuracy.append(train_acc_temp)
            '''
            pred = sess.run(prediction, feed_dict={x_data: rand_x,
                                             y_target: rand_y,
                                             prediction_grid: train_batch_x})
            print("predicted: {}".format(pred))
            print(train_batch_label)
            train_acc_temp =accuracy_score(train_batch_label, pred, normalize=True, sample_weight=None)
            train_accuracy.append(train_acc_temp)
        overall_train_accuracy.append(mean(train_accuracy))

        test_accuracy =[]
        test_loss_vec = []
        start_z=0

        b = test_index
        #a = np.asarray(a, dtype=np.int32)
        #random.shuffle(b)

        for k in range(1):
            print(k)
            test_batch_index = b[start_z:2000*(k+1)]
            #print(test_batch_index)
            #train_batch_index = np.asarray(train_batch_index, dtype=np.int32)
            #random.shuffle(train_batch_index)
            #print(train_batch_index)
            start_z=2000*(k+1)
            test_batch_x = x_vals_reduced[test_batch_index]
            #np.savetxt("train_batch_x.csv", x_vals_train, delimiter=",")
            test_batch_y = y_vals[:, test_batch_index]
            test_batch_label=label_new[test_batch_index]
            '''
            test_acc_temp = sess.run(accuracy, feed_dict={x_data: test_batch_x,
                                                    y_target: test_batch_y,
                                                    prediction_grid: test_batch_x})
            test_accuracy.append(test_acc_temp)
            '''
            pred = sess.run(prediction, feed_dict={x_data: rand_x,
                                             y_target: rand_y,
                                             prediction_grid: test_batch_x})
            #print("predicted: {}".format(pred))
            test_acc_temp =accuracy_score(test_batch_label, pred, normalize=True, sample_weight=None)
            test_accuracy.append(test_acc_temp)
        print('testing accuracy')
        print(test_accuracy)
        
        overall_test_accuracy.append(mean(test_accuracy))

print('overall train accuracy')
print(mean(overall_train_accuracy))
print('overall train standard deviation')
print(np.std(overall_train_accuracy))
print('overall test accuracy')
print(mean(overall_test_accuracy))
print('overall test standard deviation')
print(np.std(overall_test_accuracy))


