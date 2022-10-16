import glob
from turtle import title
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


X = []
Y = []
count = 1
for number in range (10):
    
    label = str (number)
    
    for img_dir in glob.glob('dataset/trainingSet/'+label+'/*.jpg'):
        # print("count", count, "label", label, img_dir)
        img = cv2.imread(img_dir)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        count += 1
        gray = gray.flatten()
        # gray.reshape(1,-1)
        X.append(np.array(gray))
        Y.append(number)
X = np.array(X)
# X.reshape(-1,1)
# print("phan tu X:", X[1])
# X.flatten()
# Y = np.array(Y)
# Y.flatten()
#splitting the dataset into 80% training data and 20% testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
# print("training size",Y_train)
# cv2.imshow("img", X_train[1])
# cv2.waitKey(0)


#building KNN model
KNN_model = KNeighborsClassifier(n_neighbors=5, p = 2, weights = 'distance')
KNN_model.fit(X_train, Y_train)

#prediction on testing data
Y_predict = KNN_model.predict(X_test)

#caculation accuracy

print("Y prediction:", Y_predict)
print("Y true: ", Y_test)

accuracy = accuracy_score(Y_predict, Y_test)
# accuracy = 100-np.mean(np.abs((Y_test-KNN_predict)/Y_test))*100

print("accuracy", accuracy)

# for i in range(200):
#     if Y_predict[i] != Y_test[i]:
#         title = "true: " +str(Y_test[i]) + "   predict: " + str(Y_predict[i]) 
#         # plt.figure()
#         plt.imshow((X_test[i].reshape(28,28)), cmap = "gray")
#         plt.title(title)
#         plt.show()
