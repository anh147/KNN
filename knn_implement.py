import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

class KNN:
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    @staticmethod
    def distance(x1, x2):
        return np.linalg.norm(x1 - x2)

    def predict_batch(self, X):
        y_pred = [self.predict(x) for x in X]
        return y_pred

    def predict(self, x):
        # Compute distance to all points in train set
        distances = [self.distance(x, x_train) for x_train in self.X_train]
        # Sort the distance with index
        top_idx = np.argsort(distances)[:self.top_k]
        # Get top K label
        k_nearests = self.y_train[top_idx]
        # Predict the label
        label = Counter(k_nearests).most_common(1)[0][0]
        
        return label
#Load data
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
    
        X.append(np.float64(gray))
        Y.append(number)
X = np.float64(X)
Y = np.array(Y)

#split data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

#building KNN model
KNN_model = KNN()
KNN_model.fit(X_train, Y_train)

#prediction on testing data
Y_predict = KNN_model.predict_batch(X_test)

#caculation accuracy

print("Y prediction:", Y_predict)
print("Y true: ", Y_test)

accuracy = accuracy_score(Y_predict, Y_test)
# accuracy = 100-np.mean(np.abs((Y_test-KNN_predict)/Y_test))*100

print("accuracy", accuracy)