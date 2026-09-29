import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import joblib

#Many more things in the scikit documentation - https://scikit-learn.org/stable/user_guide.html

digits=load_digits()
#print(digits.DESCR)

#print(digits.data)

#print(digits.data.shape)

#print(digits.feature_names)

#print(digits.target)

#print(digits.target_names)

#example = digits.data[100,:]
#example = np.reshape(example, (8,8))

#plt.imshow(example, cmap='gray')
#plt.show()

#print(digits.target[100])

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42) #Separate the data into training and test for model evaluation

#print(X_train.shape)
#print(X_test.shape)

#print(y_train.shape)
#print(y_test.shape)

best_score = 0.0
best_k = None
best_model = None

scores_list = []

for k in range(1, 21):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)

    y_pred=knn.predict(X_test)

    #print(y_pred)

    #print(y_test[0:20])
    #print(y_pred[0:20])

    scores=metrics.accuracy_score(y_test,y_pred)
    scores_list.append(scores)
    
    if(scores > best_score):
        best_score = scores 
        best_k = k
        best_model = knn
        
    print("For k = " + str(k) + ": " + str(scores))

print("The best k is " + str(best_k) + " with an accuracy of: " + str(best_score))

#save a model from scikit
if best_model is not None:
    joblib.dump(knn, "model_kNN.pkl")

#import a saved model

#knn2 = joblib.load("model_kNN.pkl")
#y_pred2=knn2.predict(X_test)

#scores=metrics.accuracy_score(y_test,y_pred)
#print(scores)

"""# Diagnostic Test

1) Repeat the training and evaluation of the kNN algorithm, but now testing different values for the number of neighbors.

2) Plot the accuracy for the test set together with another metric of your choice - check scikit for evaluation metrics. Save the figure.

3) Select the optimal number of neighbors, save the correspoding model in a pkl as shown above and submit on Fénix a zip with the following folders:

*   **figures** - containing the generated figure
*   **training_code** - this folder must contain all the code you developed during your experiments to train and select the best number of neighbors
*  **best_model** - this folder must contain:

    - a my_model.py file that contains and applies the predict function to a dataset and returns the predictions
    
    - the pickle with the best model (model.pkl)
    - aux.py with additional functions that your model may need


"""