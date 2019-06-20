import numpy as np 
import cv2 as cv
import os 
import glob
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Train set path
train_set_path = "../../dataset/train/"
train_set_label_path = "../../dataset/train/label/"
# Test set path
test_set_path = "../../dataset/test/"
test_set_label_path = "../../dataset/test/label/"

# HOG 
image_size = (64,128)
n_features = 3780
# image_size = (128,256)
# n_features = 578340
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
	print("Playing Card with HOG Classifier")
	kf = KFold(n_splits=N_FOLD, shuffle=True)
	hog = cv.HOGDescriptor()

	train_label_file = open(train_set_label_path+"labels.txt","r")
	file_contents = train_label_file.readlines()
	print("Total Files :", len(file_contents))

	print("Preparing Train Dataset From HOG Features")
	X_data = np.zeros((1,n_features), dtype=float)
	list_label = []
	for content in file_contents:
		filename, label = content.rstrip().split(",")	
		list_label.append(get_label_index(label))
		img = cv.imread(train_set_path+filename+".jpg")
		img_small = cv.resize(img, image_size)
		h = hog.compute(img_small)
		# print(h.shape)
		X_data = np.vstack((X_data, h.reshape(1,n_features)))
		# cv.imshow("Image Small", img_small)
		# cv.waitKey(1)
		print("Get HOG features from file", filename)
	X_data = np.delete(X_data, 0, 0)
	y_data = np.array(list_label)
	
	print("X_data : \n", X_data)
	print("Y_data : \n", y_data)

	kf.get_n_splits(X_data)
	rsvm_clf = svm.SVC(kernel='rbf', gamma='scale')
	lsvm_clf = svm.SVC(kernel='linear')

	list_rsvm_train_acc = []
	list_rsvm_valid_acc = []
	list_lsvm_train_acc = []
	list_lsvm_valid_acc = []
	print("Training Classifier With Train Dataset")
	print("Please Wait . . . ")
	for train_index, valid_index in kf.split(X_data):
		X_train, X_valid = X_data[train_index], X_data[valid_index]
		y_train, y_valid = y_data[train_index], y_data[valid_index]
		# SVM Classifier
		rsvm_clf.fit(X_train, y_train)		
		y_train_pred = rsvm_clf.predict(X_train)
		y_valid_pred = rsvm_clf.predict(X_valid)
		train_acc = accuracy_score(y_train, y_train_pred)
		valid_acc = accuracy_score(y_valid, y_valid_pred)
		list_rsvm_train_acc.append(train_acc)
		list_rsvm_valid_acc.append(valid_acc)
		# SVM Linear Classifier
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

	# Test C
	print("Testing Classifier With Test Image : ")
	test_label_file = open(test_set_label_path+"labels.txt","r")
	file_contents = test_label_file.readlines()
	rsvm_count_true = 0
	rsvm_count_false = 0
	lsvm_count_true = 0
	lsvm_count_false = 0
	for content in file_contents:
		filename, label = content.rstrip().split(",")	
		img = cv.imread(test_set_path+filename+".jpg")
		img_small = cv.resize(img, image_size)
		h = hog.compute(img_small)

		# Testing with SVM
		rsvm_pred = rsvm_clf.predict(h.reshape(1,n_features))
		print("Image File :", filename+".jpg", "SVM RBF Predicted :", get_label_name(int(rsvm_pred)), " True Label :", label)
		if get_label_name(int(rsvm_pred)) != label:
			print("False Detection")
			print("Close Image to Continue . . .")
			cv.imshow("Image", img)
			cv.waitKey(0)
			rsvm_count_false += 1
		else:
			print("True Detection")
			rsvm_count_true += 1

		# Testing with SVM Linear
		lsvm_pred = lsvm_clf.predict(h.reshape(1,n_features))
		print("Image File :", filename+".jpg", "SVM Linear Predicted :", get_label_name(int(lsvm_pred)), " True Label :", label)
		if get_label_name(int(lsvm_pred)) != label:
			print("False Detection")
			print("Close Image to Continue . . .")
			font = cv.FONT_HERSHEY_SIMPLEX
			cv.putText(img,label,(100,100), font, 4,(255,255,255),2,cv.LINE_AA)
			cv.imshow("Image", img)
			cv.waitKey(0)
			lsvm_count_false += 1
		else:
			print("True Detection")
			lsvm_count_true += 1

	rsvm_acc = rsvm_count_true / (rsvm_count_false + rsvm_count_true)
	print("SVM RBF Train Accuracy : ", np.sum(list_rsvm_train_acc)/N_FOLD)
	print("SVM RBF Valid Accuracy : ", np.sum(list_rsvm_valid_acc)/N_FOLD)
	print("SVM RBF Test Accuracy : ", rsvm_acc)
	lsvm_acc = lsvm_count_true / (lsvm_count_false + lsvm_count_true)
	print("SVM Linear Train Accuracy : ", np.sum(list_lsvm_train_acc)/N_FOLD)
	print("SVM Linear Valid Accuracy : ", np.sum(list_lsvm_valid_acc)/N_FOLD)
	print("SVM Linear Test Accuracy : ", lsvm_acc)

if __name__ == "__main__":
    main()