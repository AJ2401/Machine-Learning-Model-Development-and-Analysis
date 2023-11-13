# Coding K-nearest Algorithm
import time
import pandas as pd
import numpy as np
# The Lib Directly provides Function for use.
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

cols = ["fLength", 'fWidth', 'fSize', 'fConc', "fConc1",
        "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

df = pd.read_csv("Program_1/magic+gamma+telescope/magic04.data", names=cols)


df['value'] = (df['class'] == 'g').astype(int)

print(df["value"].head())

train, valid, test = np.split(
    df.sample(frac=1), [(int)(0.6*len(df)), int(0.8*len(df))])

# print("\n The Training Set \n")
# print("\t\t--------------------------------\t\t\n")
# print(train)
# print("\n The Validating Set \n")
# print("\t\t--------------------------------\t\t\n")
# print(valid)
# print("\n The Testing Set \n")
# print("\t\t--------------------------------\t\t\n")
# print(test)


def accuracy(y_test, y_pred) -> int:
    miss = 0
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            miss += 1

    return miss


def scaling(df, oversampling=False):
    x = df[df.columns[:-2]].values
    y = df[df.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversampling:

        ros = RandomOverSampler()

        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (len(y), 1))))

    return data, x, y


Train, x_train, y_train = scaling(train, oversampling=True)
Valid, x_valid, y_valid = scaling(valid, oversampling=False)
Test, x_test, y_test = scaling(test, oversampling=False)
print("\n\n\n TRAINING : \n\n\n")
print(np.sum(y_train == 1))
print(np.sum(y_train == 0))
print("\n\n\n VALIDATION : \n\n\n")
print(np.sum(y_valid == 1))
print(np.sum(y_valid == 0))
print("\n\n\n TESTING : \n\n\n")
print(np.sum(y_test == 1))
print(np.sum(y_test == 0))

# From now we will code K-Nearnest Algo

kn_model = KNeighborsClassifier(n_neighbors=6)
# Taking k value as 2
# now i am calling the fit function which will basically train or set the data in the model
kn_model.fit(x_train, y_train)
# Now i will call predict() function where i will pass the test data set to test the model.
y_pred = kn_model.predict(x_test)

print("The Original Y Test Values : ")
print(y_test)
print("The Predicted  Y Test Values : ")
print(y_pred)
# Just made a function for checking the efficiency of the program
print("\n\n\nTesting the Efficiency of the Model :\n\n\n")
st1 = time.time()
miss_values = accuracy(y_test, y_pred)
miss_percent = ((miss_values)/(len(y_test)))*100
accuracy_percent = 100-miss_percent
print("\n\n\n-------- OWN FUNCTION -----------\n\n\n")
print(
    f"The Accuracy is : {accuracy_percent} % \nThe Miss Percentage is : {miss_percent} \nNumber of Misses are : {miss_values} / {len(y_pred)}\n")
et1 = time.time()

# The Report that skLearn provides (Library function)
st2 = time.time()
print("\n\n\n-------- LIBRARAY FUNCTION -----------\n\n\n")
print(classification_report(y_test, y_pred))
et2 = time.time()

# Here i am comparing the Time Taken for both of the functions .
print("The Execution Time of\n| OWN FUNCTION  \t| LIBRARY FUNCTION \t|\n")
print(f"| {(et1-st1)*10**3 } ms | {(et2-st2)*10**3 } ms |\n\n")
