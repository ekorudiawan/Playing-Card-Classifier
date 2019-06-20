# Playing-Card-Classifier

## Dependencies

* Python 3.7
* OpenCV 3.4.6
* scikit-learn

## List Source Code

* ./sources/Image_Augmentation_Tool/Image_Augmentation_Tool.py : Source code for generating syntetic dataset from original image
* ./sources/HOG_Classifier/HOG_Classifier.py : Source code for classify the image using HOG feature detector
* ./sources/ORB_Classifier/ORB_Classifier.py : Source code for recognize the image using ORB feature detector

## List Dataset

* ./original_images : This is a directory that contains original image of 52 type playing card
* ./dataset/train : This is a directory that contains train images that was generated by source code Image_Augmentation_Tool.py
* ./dataset/test : This is a directory that contains test images that was generated by source code Image_Augmentation_Tool.py

## How to Run The Program

### Generating Synthetic Image

```bash
cd ./sources/Image_Augmentation_Tool
python Image_Augmentation_Tool.py
```

### Run Card Detector using HOG Feature

```bash
cd ./sources/HOG_Classifier
python HOG_Classifier.py
```

### Run Card Detector using ORB Feature

```bash
cd ./sources/ORB_Classifier
python ORB_Classifier.py
```
