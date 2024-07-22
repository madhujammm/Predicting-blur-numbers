
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import load_digits

# %matplotlib inline
digits = load_digits()

print("image data shape",digits.data.shape)
print("image label shape",digits.target.shape)

#if you want to change amount then sidha 5 ke jagah jitna chaiye utna daalo haar jagah
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for index, (image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
 plt.subplot(1,5,index+1)
 plt.imshow(np.reshape(image,(8,8)), cmap=plt.cm.gray)
 plt.title('training:%i\n'% label,fontsize = 20)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(x_test.shape)

from sklearn.linear_model import LogisticRegression

'''
LogisticRegr = LogisticRegression()
LogisticRegr.fit(x_train, y_train)
'''

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming you have your data loaded and split into x_train, x_test, y_train, y_test

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Increase max_iter
LogisticRegr = LogisticRegression(max_iter=1000)  # You can adjust the number of iterations as needed
LogisticRegr.fit(x_train_scaled, y_train)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Assuming you have your data loaded into digits.data and digits.target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create and train the logistic regression model
LogisticRegr = LogisticRegression()
LogisticRegr.fit(x_train_scaled, y_train)

# Now you can make predictions on the test set
prediction = LogisticRegr.predict(x_test_scaled[0].reshape(1, -1))
print(prediction)

LogisticRegr.predict(x_test[0:10])

predictions = LogisticRegr.predict(x_test)
print(predictions)

score = LogisticRegr.score(x_test,y_test)
print(score)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
cm = metrics.confusion_matrix(y_test,predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True,fmt=".3f",linewidths=.5,square = True,cmap = 'Blues_r')
plt.ylabel('actual label');
plt.xlabel('predicted label');
all_sample_title = 'accuracy score: {0}'.format(score)
plt.title(all_sample_title,size = 15);

index = 0
misclassifiedIndex = []
for predict, actual in zip(predictions, y_test):
    if predict == actual:
        misclassifiedIndex.append(index)
    index += 1

plt.figure(figsize=(20, 3))
for plotIndex, wrong in enumerate(misclassifiedIndex[0:4]):
    plt.subplot(1, 4, plotIndex + 1)
    plt.imshow(np.reshape(x_test[wrong], (8, 8)), cmap=plt.cm.gray)
    plt.title("predicted: {}, Actual: {}".format(predictions[wrong], y_test[wrong]), fontsize=20)

plt.show()
