from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import svm
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
import joblib

digits = load_digits()

X = digits.data
y = digits.target

# split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

def knn_model():
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print("accuracy of knn: ", accuracy_score(y_test,y_pred))

    # joblib.dump(knn, 'knn_model.pkl')

def svm_model():
    svm_model = svm.SVC()
    svm_model.fit(X_train, y_train)

    svm_pred = svm_model.predict(X_test)
    print("accuracy of svm: ", accuracy_score(y_test,svm_pred))

    # joblib.dump(knn, 'svm_model.pkl')


# print(digits.data.shape)
# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()