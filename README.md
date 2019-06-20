# Face Classification and Detection on FDDB

Face Detection based on classification by sliding widows

Four classifiers have been implemented:

- [x] svm
  - [x] linear kernel
  - [x] rbf kernel
  - [x] sigmoid kernel
- [x] logistic 
  - [x] sgd
  - [x] langevin
- [x] fisher
- [x] cnn

CNN will extract features from original images, and other classifiers will use HOG features to classify. The detector is based on face classification and the detection process is:

+ generate sliding windows with all kinds of sizes
+ crop subimages according to windows and resize to 96x96
+ use the pretrained classifier to predict
+ select the top 30 and use NMS to obtain the last result

## Setup

### Download

Download the FDDB [dataset](http://vis-www.cs.umass.edu/fddb/)  including original images and annotation, and extract and do:

```
mkdir dataset && cd dataset
mkdir -p h5 hog cropPicsPos/list cropPicsPos/list
mv -r FDDB-folds dataset/
mv -r originalPics dataset/
```

### Install

```
conda create -n detection python=3
conda activate detection
conda install -c conda-forge opencv
conda install -c anaconda scikit-image
conda install -c anaconda scikit-learn 
conda install pytorch
```



## Preprocess

### generate positive samples

`python generate_negative.py`

+ Use a rectangle without any rotations

+ extend the scale of the bounding box by 1/3

+ Crop the face within the bounding box as a positive sample

![image](https://github.com/MondayYuan/FaceClassifcationAndDetection/blob/master/fig/pos.png)

### generate negative samples

`python generate_positve.py`

generate eight negative images based on each face by sliding the bounding box by 1/3

![image](https://github.com/MondayYuan/FaceClassifcationAndDetection/blob/master/fig/neg.png)


### generate HOG features

`python generate_hog.py`

+ Resize all samples (including clipped faces and negative samples) into 96  x 96 

+ Extract HOG features from each sample using skimage

### generate h5 file

`python generate_h5.py`

CNN takes h5 files as input

### visualization

```
python visualize_box.py
python visualize_hog.py
```



# Classify

## train 

```
python main.py classify train --model=cnn --save-path=log/cnn.pkl
python main.py classify train --model=fisher --save-path=log/fisher.pkl
python main.py classify train --model=svm --kernel=linear --save-path=log/svm-linear.pkl
python main.py classify train --model=logistic --solver=sgd --save-path=log/logistic.pkl
```

## test

```
python main.py classify test --model=cnn --load-path=log/cnn.pkl --image=test.jpg
```

# Detect

```
python main.py detect --model=cnn --load-path=log/cnn.pkl --image=test.jpg --save-image=result.jpg
```

