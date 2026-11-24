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
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score # or Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split #cross_val_score is used to evaluate the performance of a model using cross-validation, and train_test_split is used to split the data into training and testing sets
import librosa
from sklearn.svm import SVC #i probably wont need this but just in case i do
from sklearn.calibration import CalibratedClassifierCV
import joblib




#0. i dont actually have any data to work with, so ill just make some random data to test the code with
SRC_DIR = Path(__file__).resolve().parent #finds the src/ folder where this script lives
DATA_DIR = SRC_DIR / "data" / "processed" #points to src/data/processed/


clips = np.load(DATA_DIR / "clips_16k.npy").astype(np.float32) #loads the audio clips and converts them to float32 for librosa
labels = pd.read_csv(DATA_DIR / "labels.csv") #this is the labels for the clips, it contains the chord labels for each clip, and the index of the clip in the clips array


chroma = librosa.feature.chroma_stft(y=clips, sr=16000, n_fft=8192, hop_length=4096) #this is the chroma features for the clips, it contains the 12 chroma bins across time frames for each clip


X = np.concatenate([chroma.mean(axis=-1), chroma.std(axis=-1)], axis=1) #this is the input features for the model, it contains the mean and standard deviation of the 12 chroma features (24 total) for each clip
y = labels["original_label"].str.split("/").str[0].to_numpy() #this is the target labels for the model, it contains the original chord labels for each clip, and the index of the clip in the clips array


#1. Split the data into training and testing sets (70% train / 15% val / 15% test)


X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42) #ive split the data into training and remaining sets, holding 30% aside for val/test and using a random state of 42 (because chatgpt told me to)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, stratify=y_rem, random_state=42) #splits that remaining 30% evenly into 15% validation and 15% test


scaler = StandardScaler() #i will use this to scale the features before feeding them into the models, this is important because some models are sensitive to the scale of the features
scaler.fit(X_train) #learns the average and srandard deviation from the training data
X_train_sc = scaler.transform(X_train)
X_val_sc = scaler.transform(X_val) #scales the validation set using the training mean and std
X_test_sc = scaler.transform(X_test)

MODEL_DIR = SRC_DIR / "models" #points to src/models/ where saved models will go
MODEL_DIR.mkdir(exist_ok=True) #creates the src/models/ folder if it doesnt exist yet
joblib.dump(scaler, MODEL_DIR / "scaler.joblib") #saves the fitted scaler so presentation_models.py can scale live clips the exact same way


#2. Tune k-NN on the validation set before testing
Kvals = [1, 3, 5, 9, 15] #candidate values for k (number of neighbours)
Kaccuracies = [] #stores validation accuracy for each k


for k in Kvals:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_sc, y_train) #fit k-NN on the scaled training data
    val_pred = knn.predict(X_val_sc) #test this k on the validation set
    Kaccuracies.append(accuracy_score(y_val, val_pred))


best_k = Kvals[np.argmax(Kaccuracies)] #picks the k that got the highest validation accuracy
print(f"Best k for k-NN: {best_k} (val accuracy: {max(Kaccuracies):.3f})\n")


models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "k-NN": KNeighborsClassifier(n_neighbors=best_k),
    "SVM": CalibratedClassifierCV(SVC(C=10), ensemble=False),#C=10 improves accuracy and probability=True will let the demo show confidence bars later [note that i had to use CalibratedClassifierCV because of some update thingy
    "Random Forest": RandomForestClassifier(random_state=42)
}   #here ive made a dictionary to hold the two models, with their names as keys and pipelines as values, this will make it easier to loop through them for evaluation
#now the basic models are complete, i will now attempt to implement SVM and random forest classifiers, and compare their performace with the previous modeles


filenames = {
    "Logistic Regression": "logreg.joblib",
    "k-NN": "knn.joblib",
    "SVM": "svm.joblib",
    "Random Forest": "rf.joblib"
} #maps each model name to its filename in src/models/


for name, model in models.items():
    model.fit(X_train_sc, y_train) #fit the model to the training data
    y_pred = model.predict(X_test_sc) #predict the labels for the test data
    accuracy = accuracy_score(y_test, y_pred) #calculate the accuract of the model, this is done by comparing the predicted labels with the true labels for the test data
    print(f"{name}: {accuracy:.3f}")
    joblib.dump(model, MODEL_DIR / filenames[name]) #saves the trained model to src/models/








#the current format of the labels is "C:maj/3", i need to convert them to 2 seperate plain lists
#this means i can train the models on the root and quality seperately


y_train_root = [label.split(":")[0] for label in y_train] #the features are already in the correct format (train_test_split) so i just need to split them
y_train_quality = [label.split(":")[1].split("/")[0] for label in y_train] #the quality


y_test_root = [label.split(":")[0] for label in y_test] #same for the test set
y_test_quality = [label.split(":")[1].split("/")[0] for label in y_test] #same as above






randomforest_root = RandomForestClassifier(random_state=42)
randomforest_quality = RandomForestClassifier(random_state=42) #random bullshit to make sure the models are working


randomforest_root.fit(X_train_sc, y_train_root) #trains the root model
randomforest_quality.fit(X_train_sc, y_train_quality) #trains the quality model


#seperate predictions:
root_pred = randomforest_root.predict(X_test_sc) #predict the root labels for the test data
quality_pred = randomforest_quality.predict(X_test_sc) #predict the quality labels for the test
combined_pred = [r + ":" + q for r, q in zip(root_pred, quality_pred)]  #zip just pairs the items up




#just outputing the accuracty scored for each model seperate and a combined one
print(f"root accuracy: {accuracy_score(y_test_root, root_pred):.3f}")
print(f"quality accuracy: {accuracy_score(y_test_quality, quality_pred):.3f}")
print(f"combined: {accuracy_score(y_test, combined_pred):.3f}")






# DONE:
# - Loaded 16 kHz audio clips and simplified labels to 84 chord classes (root:quality).
# - Extracted 24 chroma features (12 means + 12 stds) and split 70/15/15 (train/val/test).
# - Fitted StandardScaler on train, tuned k-NN on val, and calibrated SVM.
# - Evaluated all 4 single models and tested the 2-model (root + quality) Random Forest setup.
# - Saved the fitted scaler and all 4 trained models to src/models/ using joblib.dump.
#
# TO DO:
# 1. Register the models in src/presentation_models.py using @Model (load scaler + model,
#    extract the exact same 24 chroma features, scale, and return label + confidence).
# 2. Launch presentation.py and test all models in the live demo grid.
# 3. Commit and push stage4.py and presentation_models.py to GitHub and open a PR.
# 4. Document the dataset, validation tuning, 1-model vs. 2-model results, and test scores in the project report.
