import numpy as np 
import cv2 as cv
import os 
import glob
from sklearn import svm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

dataset_path = "/home/Playing-Card-Classifier/dataset/"
label_path = "/home/Playing-Card-Classifier/dataset/label/"

n_clusters = 100
labels = []
N_FOLD = 10

label_names = ['10C', '10D', '10H', '10S', \
                '2C', '2D', '2H', '2S', \
                '3C', '3D', '3H', '3S', \
                '4C', '4D', '4H', '4S', \
                '5C', '5D', '5H', '5S', \
                '6C', '6D', '6H', '6S', \
                '7C', '7D', '7H', '7S', \
                '8C', '8D', '8H', '8S', \
                '9C', '9D', '9H', '9S', \
                'AC', 'AD', 'AH', 'AS', \
                'JC', 'JD', 'JH', 'JS', \
                'KC', 'KD', 'KH', 'KS', \
                'QC', 'QD', 'QH', 'QS']

def get_label_index(label):
    return label_names.index(label)

def get_label_name(index):
	return label_names[index]

def main():
    print("Playing Card with ORB Classifier")
    kf = KFold(n_splits=N_FOLD, shuffle=True)
    orb = cv.ORB_create(200)
    list_descriptor = []
    label_file = open(label_path+"labels.txt","r")
    file_contents = label_file.readlines()
    print("Total Files :", len(file_contents))

    # Get ORB descriptor from each image
    print("Get ORB descriptor")
    for content in file_contents:
        filename, label = content.rstrip().split(",")
        
        img = cv.imread(dataset_path+filename+".jpg")
        _, des = orb.detectAndCompute(img, None)
        list_descriptor.append(des)
    
    # Combine all descriptor
    descriptor_stack = np.zeros((1,32), dtype=np.uint8)
    for descriptor in list_descriptor:
        m, _ = descriptor.shape
        for i in range(m):
            descriptor_stack = np.vstack((descriptor_stack, descriptor[i]))
    descriptor_stack = np.delete(descriptor_stack, 0, 0)

    # Clustering to generate bag of visual word from all descriptor
    print("Train KMeans")
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(descriptor_stack)

    # Create feature histogram for each image
    print("Create feature histogram")
    img_histogram = np.zeros((len(file_contents),n_clusters), dtype=np.uint8)
    counter = 0
    for content in file_contents:
        filename, label = content.rstrip().split(",")       
        img = cv.imread(dataset_path+filename+".jpg")
        _, des = orb.detectAndCompute(img, None)
        for i in range(len(des)):
            idx = kmeans.predict(des[i].reshape(1, 32))
            img_histogram[counter,idx] = img_histogram[counter,idx] + 1
        counter += 1
    
    # Normalize feature histogram
    print("Normalize feature histogram")
    scaler = StandardScaler().fit(img_histogram)
    img_histogram = scaler.transform(img_histogram)

    # Create dataset from feature histogram
    print("Create dataset from feature histogram")
    counter = 0
    list_label = []
    X_data = np.zeros((len(file_contents),n_clusters), dtype=np.float32)
    for content in file_contents:
        filename, label = content.rstrip().split(",")
        X_data[counter] = img_histogram[counter]
        counter += 1
        list_label.append(get_label_index(label))
    y_data = np.array(list_label)

    print("Training dataset")
    kf.get_n_splits(X_data)
    clf = svm.SVC(gamma='scale')
    counter = 0
    list_train_acc = []
    list_valid_acc = []
    for train_index, valid_index in kf.split(X_data):
        X_train, X_valid = X_data[train_index], X_data[valid_index]
        y_train, y_valid = y_data[train_index], y_data[valid_index]
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_valid_pred = clf.predict(X_valid)
        train_acc = accuracy_score(y_train, y_train_pred)
        valid_acc = accuracy_score(y_valid, y_valid_pred)
        list_train_acc.append(train_acc)
        list_valid_acc.append(valid_acc)
        counter += 1
    print("Train Accuracy : ", np.sum(list_train_acc)/N_FOLD)
    print("Valid Accuracy : ", np.sum(list_valid_acc)/N_FOLD)

if __name__ == "__main__":
    main()