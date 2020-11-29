# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import tensorflow as tf
#from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import csv
from sklearn.model_selection import KFold
import pylab as pl
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
  
# loading the iris dataset 
#load the dataset
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

from scipy import stats
x_vals=stats.zscore(x_vals)
#print(x_vals)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
x_vals=imp.fit_transform(x_vals)

reduction=20
pca = PCA(n_components=reduction)
principalComponents = pca.fit_transform(x_vals)
xx_vals_reduced = pd.DataFrame(data = principalComponents)
#x_vals_reduced= xx_vals_reduced.astype(np.float)
x_vals_reduced= np.squeeze(np.asarray(xx_vals_reduced))
#x_vals_reduced= (x_vals_reduced - mean(x_vals_reduced)) / std(x_vals_reduced)
#print(x_vals_reduced[1])

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
print(x_vals_reduced.shape)
print(label.shape)




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

test_accuracy =[]
train_accuracy =[]
kf = KFold(n_splits=9)
for train_index, test_index in kf.split(x_vals_reduced):
  
        # dividing X, y into train and test data 
        x_vals_train = x_vals_reduced[train_index]
        x_vals_test = x_vals_reduced[test_index]
        y_vals_train = label[train_index]
        y_vals_test = label[test_index]
  
        # training a linear SVM classifier 
        from sklearn.svm import SVC 
        svm_model = SVC(kernel = 'rbf', C = 1, gamma=16).fit(x_vals_train, y_vals_train) 
        svm_predictions = svm_model.predict(x_vals_test) 
  
        # model accuracy for X_test  
        accuracy = svm_model.score(x_vals_train, y_vals_train) 
        print(accuracy) 
        train_accuracy.append(accuracy) 

        accuracy = svm_model.score(x_vals_test, y_vals_test) 
        print(accuracy) 
        test_accuracy.append(accuracy)     
        # creating a confusion matrix 
        cm = confusion_matrix(y_vals_test, svm_predictions) 
        print(cm)


print('overall train accuracy')
print(mean(train_accuracy))
print('overall train standard deviation')
print(np.std(train_accuracy))
print('overall test accuracy')
print(mean(test_accuracy))
print('overall test standard deviation')
print(np.std(test_accuracy))

# Create a mesh to plot points in
x_min= int(min(x_vals_reduced[:, 0]))-1
x_max = int(max(x_vals_reduced[:, 0])) +1
y_min, y_max = int(min(x_vals_reduced[:, 1])) - 1, int(max(x_vals_reduced[:, 1]))+ 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                     np.arange(y_min, y_max, 0.2))
#print(xx)
grid_predictions = svm_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)


# Plot points and grid
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.4)
plt.plot(class1_x, class1_y, '1', label='1st room')
plt.plot(class2_x, class2_y, '2', label='2nd room')
plt.plot(class3_x, class3_y, '3', label='3rd room')
plt.plot(class4_x, class4_y, '4', label='4th room')
plt.title('Dataset 1_PCA=2_Gamma=1000')
plt.xlabel('First dimension')
plt.ylabel('Second dimension')
plt.legend(loc='lower right')
plt.ylim([x_min, x_max])
plt.xlim([y_min, y_max])
plt.show()

# Evaluations on new/unseen data