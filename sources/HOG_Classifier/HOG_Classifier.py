import numpy as np 
import cv2 as cv
import os 
import glob
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

dataset_path = "/home/Playing-Card-Classifier/dataset/"
label_path = "/home/Playing-Card-Classifier/dataset/label/"

# HOG 
# image_size = (64,128)
# n_features = 3780
image_size = (128,256)
n_features = 578340
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

	label_file = open(label_path+"labels.txt","r")
	file_contents = label_file.readlines()
	print("Total Files :", len(file_contents))

	print("Preparing train dataset")
	X_data = np.zeros((1,n_features), dtype=float)
	list_label = []
	for content in file_contents:
		filename, label = content.rstrip().split(",")	
		list_label.append(get_label_index(label))
		img = cv.imread(dataset_path+filename+".jpg")
		img_small = cv.resize(img, image_size)
		h = hog.compute(img_small)
		print(h.shape)
		X_data = np.vstack((X_data, h.reshape(1,n_features)))
		# cv.imshow("Image Small", img_small)
		# cv.waitKey(1)
		print("Create HOG features", filename)
	X_data = np.delete(X_data, 0, 0)
	y_data = np.array(list_label)
	print("Training dataset")
	print("X_data : \n", X_data)
	print("Y_data : \n", y_data)

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

	# print("Testing Classifier : ")
	# for content in file_contents:
	# 	filename, label = content.rstrip().split(",")	
	# 	list_label.append(get_label_index(label))
	# 	img = cv.imread(dataset_path+filename+".jpg")
	# 	img_small = cv.resize(img, image_size)
	# 	h = hog.compute(img_small)
	# 	pred = clf.predict(h.reshape(1,n_features))
		
	# 	print("Image file :", filename+".jpg", "Predicted :", get_label_name(int(pred)), " True Label :", label)
	# 	if get_label_name(int(pred)) != label:
	# 		print("False")
	# 		cv.imshow("Image", img)
	# 		cv.waitKey(0)
	# 	else:
	# 		print("True")

if __name__ == "__main__":
    main()