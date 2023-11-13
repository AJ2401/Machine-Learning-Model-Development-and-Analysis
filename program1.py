import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Making Columns
# 1.  fLength:  continuous  # major axis of ellipse [mm]
#  2.  fWidth:   continuous  # minor axis of ellipse [mm]
#  3.  fSize:    continuous  # 10-log of sum of content of all pixels [in #phot]
#  4.  fConc:    continuous  # ratio of sum of two highest pixels over fSize  [ratio]
#  5.  fConc1:   continuous  # ratio of highest pixel over fSize  [ratio]
#  6.  fAsym:    continuous  # distance from highest pixel to center, projected onto major axis [mm]
#  7.  fM3Long:  continuous  # 3rd root of third moment along major axis  [mm]
#  8.  fM3Trans: continuous  # 3rd root of third moment along minor axis  [mm]
#  9.  fAlpha:   continuous  # angle of major axis with vector to origin [deg]
# 10.  fDist:    continuous  # distance from origin to center of ellipse [mm]
# 11.  class:    g,h         # gamma (signal), hadron (background)

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym",
        "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
# Converting it into a Data Frame
df = pd.read_csv("Program_1/magic+gamma+telescope/magic04.data", names=cols)
# Displaying first 5 rows of the data frame
print(df.head())
# It will give the unique objects present in the column
# for this example we will get g -> gamma h-> hydrons
# it can be f-> female or m -> male
print("The Class Column for the Data Set \n" + str(df['class'].unique()))
#  How to check if the whole column contains certain values ?
df['value'] = (df['class'] == 'g').astype(int)
# If the whole column has only g then it will give 1 or if contains h then it will return 0
print(df)

# Here we are taking all the columns but not the output column.
for label in cols[:-1]:
    # So for each label only the value == 1 rows will come and it will make a histrogram.
    plt.hist(df[df["value"] == 1][label],
             color='blue', density=True, alpha=0.7, label="Gamma",)
    plt.hist(df[df["value"] == 0][label],
             color='red', density=True, alpha=0.5, label="Hydrons",)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    # Overlapping
    # The legend function in Python is used to place a legend on the axis of a plot or subplot.3 The function has an attribute called loc that can be used to specify the location of the legend.
    plt.legend()
    plt.show()
