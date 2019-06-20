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

# Train set path
train_set_path = "../../dataset/train/"
train_set_label_path = "../../dataset/train/label/"
# Test set path
test_set_path = "../../dataset/test/"
test_set_label_path = "../../dataset/test/label/"

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
    label_file = open(train_set_label_path+"labels.txt","r")
    file_contents = label_file.readlines()
    print("Total Files :", len(file_contents))

    print("Preparing Train Dataset From ORB Features")
    # Get ORB descriptor from each image
    print("Get ORB Features From Images")
    for content in file_contents:
        filename, label = content.rstrip().split(",")
        img = cv.imread(train_set_path+filename+".jpg")
        _, des = orb.detectAndCompute(img, None)
        list_descriptor.append(des)
        print("Get ORB features from file", filename)
    
    # Combine all descriptor
    descriptor_stack = np.zeros((1,32), dtype=np.uint8)
    for descriptor in list_descriptor:
        m, _ = descriptor.shape
        for i in range(m):
            descriptor_stack = np.vstack((descriptor_stack, descriptor[i]))
    descriptor_stack = np.delete(descriptor_stack, 0, 0)

    # Clustering to generate bag of visual word from all descriptor
    print("Clustering HOG Features to", n_clusters, "clusters")
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(descriptor_stack)

    # Create feature histogram for each image
    print("Create Feature Histogram")
    feature_histogram = np.zeros((len(file_contents),n_clusters), dtype=np.uint8)
    counter = 0
    for content in file_contents:
        filename, label = content.rstrip().split(",")       
        img = cv.imread(train_set_path+filename+".jpg")
        _, des = orb.detectAndCompute(img, None)
        for i in range(len(des)):
            idx = kmeans.predict(des[i].reshape(1, 32))
            feature_histogram[counter,idx] = feature_histogram[counter,idx] + 1
        # plt.hist(feature_histogram[counter], bins=n_clusters)
        # plt.show()
        counter += 1
        
        print("Generate Features Histogram From File", filename)
    
    # Normalize feature histogram
    print("Normalize Feature Histogram")
    scaler = StandardScaler().fit(feature_histogram)
    feature_histogram = scaler.transform(feature_histogram)

    # Create dataset from feature histogram
    print("Create Dataset From Feature Histogram")
    counter = 0
    list_train_label = []
    X_data = np.zeros((len(file_contents),n_clusters), dtype=np.float32)
    for content in file_contents:
        filename, label = content.rstrip().split(",")
        X_data[counter] = feature_histogram[counter]
        counter += 1
        list_train_label.append(get_label_index(label))
    y_data = np.array(list_train_label)

    print("Train Classifier With Train Dataset")
    kf.get_n_splits(X_data)
    rsvm_clf = svm.SVC(kernel='rbf',gamma='scale')
    lsvm_clf = svm.SVC(kernel='linear')

    list_rsvm_train_acc = []
    list_rsvm_valid_acc = []
    list_lsvm_train_acc = []
    list_lsvm_valid_acc = []
    for train_index, valid_index in kf.split(X_data):
        X_train, X_valid = X_data[train_index], X_data[valid_index]
        y_train, y_valid = y_data[train_index], y_data[valid_index]
        # SVM RBF kernel
        rsvm_clf.fit(X_train, y_train)
        y_train_pred = rsvm_clf.predict(X_train)
        y_valid_pred = rsvm_clf.predict(X_valid)
        train_acc = accuracy_score(y_train, y_train_pred)
        valid_acc = accuracy_score(y_valid, y_valid_pred)
        list_rsvm_train_acc.append(train_acc)
        list_rsvm_valid_acc.append(valid_acc)
        # SVM linear kernel
        lsvm_clf.fit(X_train, y_train)
        y_train_pred = lsvm_clf.predict(X_train)
        y_valid_pred = lsvm_clf.predict(X_valid)
        train_acc = accuracy_score(y_train, y_train_pred)
        valid_acc = accuracy_score(y_valid, y_valid_pred)
        list_lsvm_train_acc.append(train_acc)
        list_lsvm_valid_acc.append(valid_acc)

    print("SVM RBF Train Accuracy : ", np.sum(list_rsvm_train_acc)/N_FOLD)
    print("SVM RBF Valid Accuracy : ", np.sum(list_rsvm_valid_acc)/N_FOLD)
    print("SVM Linear Train Accuracy : ", np.sum(list_lsvm_train_acc)/N_FOLD)
    print("SVM Linear Valid Accuracy : ", np.sum(list_lsvm_valid_acc)/N_FOLD)

    print("Testing Classifier With Test Image : ")
    test_label_file = open(test_set_label_path+"labels.txt","r")
    file_contents = test_label_file.readlines()
    rsvm_count_true = 0
    rsvm_count_false = 0
    lsvm_count_true = 0
    lsvm_count_false = 0
    feature_histogram = np.zeros((1,n_clusters), dtype=np.uint8)
    for content in file_contents:
        filename, label = content.rstrip().split(",")
        img = cv.imread(test_set_path+filename+".jpg")
        _, des = orb.detectAndCompute(img, None)
        for i in range(len(des)):
            idx = kmeans.predict(des[i].reshape(1, 32))
            feature_histogram[0,idx] = feature_histogram[0,idx] + 1

        feature_histogram = scaler.transform(feature_histogram)

        # Predict using SVM RBF
        rsvm_pred = rsvm_clf.predict(feature_histogram.reshape(1,n_clusters))
        print("Image File :", filename+".jpg", "SVM RBF Predicted :", get_label_name(int(rsvm_pred)), " True Label :", label)
        if get_label_name(int(rsvm_pred)) != label:
            print("False Detection")
            print("Close Image to Continue . . .")
            # cv.imshow("Image", img)
            # cv.waitKey(0)
            rsvm_count_false += 1
        else:
            print("True Detection")
            rsvm_count_true += 1

        # Predict using SVM Linear
        lsvm_pred = lsvm_clf.predict(feature_histogram.reshape(1,n_clusters))
        print("Image File :", filename+".jpg", "SVM Linear Predicted :", get_label_name(int(lsvm_pred)), " True Label :", label)
        if get_label_name(int(lsvm_pred)) != label:
            print("False Detection")
            print("Close Image to Continue . . .")
            # plt.hist(feature_histogram, normed=True, bins=n_clusters)
            # plt.show()
            lsvm_count_false += 1
        else:
            print("True Detection")
            lsvm_count_true += 1

    rsvm_acc = rsvm_count_true / (rsvm_count_false + rsvm_count_true)
    lsvm_acc = lsvm_count_true / (lsvm_count_false + lsvm_count_true)
    print("SVM RBF Train Accuracy : ", np.sum(list_rsvm_train_acc)/N_FOLD)
    print("SVM RBF Valid Accuracy : ", np.sum(list_rsvm_valid_acc)/N_FOLD)
    print("SVM RBF Test Accuracy : ", rsvm_acc)
    print("SVM Linear Train Accuracy : ", np.sum(list_lsvm_train_acc)/N_FOLD)
    print("SVM Linear Valid Accuracy : ", np.sum(list_lsvm_valid_acc)/N_FOLD)
    print("SVM Linear Test Accuracy : ", lsvm_acc)

if __name__ == "__main__":
    main()