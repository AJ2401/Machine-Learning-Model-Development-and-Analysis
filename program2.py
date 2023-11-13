# Following the model Training process
# Training data set , Validation data set , Testing data set
import pandas as pd
import matplotlib as plt
import numpy as np
# class sklearn.preprocessing.StandardScaler(*, copy=True, with_mean=True, with_std=True)
# std:standard Score
# z = (x - u) / s
# x is actual value
# s is standard deviation
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

cols = ["fLength", 'fWidth', 'fSize', 'fConc', "fConc1",
        "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

df = pd.read_csv("Program_1/magic+gamma+telescope/magic04.data", names=cols)


df['value'] = (df['class'] == 'g').astype(int)

print(df["value"].head())

train, valid, test = np.split(df.sample(frac=1), [(int)(
    0.6 * len(df)), (int)(0.8 * len(df))])

print("\n The Training Set \n")
print("\t\t--------------------------------\t\t\n")
print(train)
print("\n The Validating Set \n")
print("\t\t--------------------------------\t\t\n")
print(valid)
print("\n The Testing Set \n")
print("\t\t--------------------------------\t\t\n")
print(test)

# # Here when we see the data set the values in the colums differs like one column has 100+ values and one column has value in 0.0+


def scaling(df, oversample=False):
    x = df[df.columns[:-2]].values  # All columns values
    y = df[df.columns[-1]].values  # Last col values

    # Creating a Scaler
    scaler = StandardScaler()
#     # from scaler we can do fit and transform x
    x = scaler.fit_transform(x)
    if oversample:
        ros = RandomOverSampler()
        # by this function it will take the low sample and add some random sample from the same data set.
        x, y = ros.fit_resample(x, y)
#     # Now the whole modified data will be created using 2D numpy
#     # hstack is horizontal stacks of arrays
#     # We will be modifying the y because x is a 2D matrix whereas y is array so making it into a 2D matrix
    data = np.hstack((x, np.reshape(y, (len(y), 1))))

    return data, x, y


# Calling the Scaling Function for Equallising the training data set.
Train, x_train, y_train = scaling(train, oversample=True)
# Here Oversample is False because these are Testing and validating data.
# We just want the Accuracy of the Model which can be analysed by any length of the data set
Valid, x_valid, y_valid = scaling(valid, oversample=False)
Test, x_test, y_test = scaling(test, oversample=False)
print("\n The Value Array \n")
print("\t\t--------------------------------\t\t\n")
print(y_train)
# The Gamma Data Set
print("\n The Gamma Values in Value Array \n")
print("\t\t--------------------------------\t\t\n")
print(np.sum(y_train == 1))
# The Hetrons Data Set
print("\n The Hyderons Values in Value Array \n")
print("\t\t--------------------------------\t\t\n")
print(np.sum(y_train == 0))
# Here we will get same size due to Random Over Sampler

# From this Program we can see that all our data
