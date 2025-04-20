from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
import joblib
import numpy as np
# import pandas as pd

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(np.uint8)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='rbf', C=5, gamma=0.05)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
joblib.dump(svm, 'svm_mnist_model_rbf.pkl')