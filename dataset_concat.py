import numpy as np
import pandas as pd

PATH_TO_CSV_FOLDER = "./training/"

dataset_sub1_3m = pd.read_csv(PATH_TO_CSV_FOLDER + "sub1_3m_straight.csv")
print(dataset_sub1_3m.head())
print(dataset_sub1_3m.shape)
dataset_sub1_4m = pd.read_csv(PATH_TO_CSV_FOLDER + "sub1_4m_straight.csv")
print(dataset_sub1_4m.head())
print(dataset_sub1_4m.shape)
#dataset = pd.concat([ dataset_sub1_3m, dataset_sub1_4m], ignore_index = True)
#print(dataset.shape)
dataset = pd.read_csv(PATH_TO_CSV_FOLDER + "sub1_3m_straight.csv")


for i in range(3,9):
    dataset_list = pd.read_csv(PATH_TO_CSV_FOLDER + "sub1_" + str(i) + "m_straight.csv")
    dataset = pd.concat([ dataset, dataset_list], ignore_index = True)

print(dataset)

dataset.to_csv(PATH_TO_CSV_FOLDER + "sub1_3-8.csv", index = False)

