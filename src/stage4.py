

# this script builds and tests the core training workflow for stage 4 before
# stage 3's real features (features.npy) are ready. using dummy data in the exact
# shape of the real dataset (7,398 clips x 24 chroma features, 84 chord classes), it:
#   1. holds back 20% of the data as a mock test set using train_test_split.
#   2. manually fits standardscaler on the training set and transforms both train and test features.
#   3. trains and compares four classifiers (logistic regression, k-nn, svm, and random forest)
#      using .fit(), .predict(), and accuracy_score.
# once stage 3 is done, the dummy arrays will be swapped for real data and hyperparameter
# tuning will be added.



#1. The inputs i should receive are X and y
# X is a 2D array of shape (n_samples, n_features) containing the input features for each sample
#y is a 1D array of shape (n_samples,) containing the target labels for each sample

#2. I need to split the data into training and testing sets. I will use sklearn's train_test_split function for this purpose. I will specify the test size and random state for reproducibility.
# ill hold 20% of the data for testing and use a random state of 42 for reproducibility.

#3. I will scale the 24 features manually with StandardScaler, then store four classifiers in a dictionary:
# - Logistic Regression
# - k-NN
# - SVM
# - Random Forest

#4 i will evaluate the performance of all four models by training them on the scaled training set (.fit()), predicting on the scaled test set (.predict()), and computing accuracy with accuracy_score.

#-----imports-------
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score # or Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split #cross_val_score is used to evaluate the performance of a model using cross-validation, and train_test_split is used to split the data into training and testing sets
import librosa
from sklearn.svm import SVC #i probably wont need this but just in case i do


#0. i dont actually have any data to work with, so ill just make some random data to test the code with
X = np.random.rand(7398, 24)  # 7398 samples, 24 features
y = np.random.randint(0, 84, size=7398)  # 84 chord classes (0 to 83) for 7398 samples

#1. Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #ive split the data into training and testing sets, holding 20% of the data for testing and using a random state of 42 (because chatgpt told me to)

scaler = StandardScaler() #i will use this to scale the features before feeding them into the models, this is important because some models are sensitive to the scale of the features
scaler.fit(X_train) #learns the average and srandard deviation from the training data
X_train_sc = scaler.transform(X_train) 
X_test_sc = scaler.transform(X_test) 

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "k-NN":KNeighborsClassifier(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier() 
}   #here ive made a dictionary to hold the two models, with their names as keys and pipelines as values, this will make it easier to loop through them for evaluation
#now the basic models are complete, i will now attempt to implement SVM and random forest classifiers, and compare their performace with the previous modeles


for name, model in models.items():
    model.fit(X_train_sc, y_train) #fit the model to the training data
    y_pred = model.predict(X_test_sc) #predict the labels for the test data
    accuracy = accuracy_score(y_test, y_pred) #calculate the accuract of the model, this is done by comparing the predicted labels with the true labels for the test data 
    print(f"{name}: {accuracy:.3f}")
