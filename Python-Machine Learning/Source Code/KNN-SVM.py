#Question 2: Create k-nearest Neighbors, Support Vector Machines (SVM), and a  Naive Bayes classifier to classify handwritten digits from the MNIST datasetLinks to an external site.. Implement these #models in Python using scikit-learn. (10 points)

#Load the MNIST dataset using the scikit-learn dataset module.
#Split the dataset into training and testing sets.
#Implement the classifiers using scikit-learn.
#Train the model on the training set.
#Evaluate the model's performance using appropriate metrics.
#Include comments to explain each step of your code.

#Script-Assignment-part-2
from sklearn import datasets, neighbors, svm, naive_bayes
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


#Loads the dataset
mnist = datasets.load_digits()
pd = datasets.load_digits()
dataset = list(zip(mnist.images,mnist.target))
print('Dataset: ', dataset)

#Splits into training and testing data
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.25)
print('Training Data Shape: ', X_train.shape)
print('Test Data Shape: ', X_test.shape)

##Implements classifiers

#Nearest k-neighbor classifier
knn = neighbors.KNeighborsClassifier()

#Train the KNN model on the training dataset
knn.fit(X_train, y_train)

#Evaluates performance of KNN on the test dataset
y_pred_knn = knn.predict(X_test)
print("Accuracy Score for KNN = %2f%%" % (accuracy_score(y_test, y_pred_knn)*100))


#Support Vector Machine classifier
svm = svm.SVC()

#Trains the SVM  on the training dataset
svm.fit(X_train, y_train)

#Evaluates performance of SVM on the test dataset
y_pred_svm = svm.predict(X_test)
print("Accuracy Score for SVM = %2f%%" % (accuracy_score(y_test, y_pred_svm)*100))

#Naive Bayes Classifier
gnb = naive_bayes.GaussianNB()

#Trains the GNB  on the training dataset
gnb.fit(X_train, y_train)

#Evaluates performance of SVM on the test dataset
y_pred_gnb = gnb.predict(X_test)
print("Accuracy Score for GaussaianNB = %2f%%" % (accuracy_score(y_test, y_pred_gnb)*100))


#Displays confusion matrix & classification report for the classifiers
def evaluate_model(y_true, y_pred, model_name):

    #Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm)
    plt.title('Confusion Matrix ' + model_name)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    report = classification_report(y_true, y_pred)
    print(f'Classification Report - {model_name}:\n{report}')

#Evaluates KNN model with the confusion matrix
evaluate_model(y_test, y_pred_knn, 'KNN')

#Evaluates SVM model with the confusion matrix
evaluate_model(y_test, y_pred_svm, 'SVM')

#Evaluates GNB model with the confusion matrix
evaluate_model(y_test, y_pred_gnb, 'GaussianNB')