import numpy as np 
import pandas as pd  

PATH_TO_TRAINING_FOLDER = "./training/"

dataset = pd.read_csv(PATH_TO_TRAINING_FOLDER + "sub1_3-8.csv")
print("Shape of dataset = ", dataset.shape)
print("No of feature vectors in  dataset = ", dataset.shape[0])
print("Shape of columns in dataset = ", dataset.shape[1])