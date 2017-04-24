# AutomatedEmbryoPrediction

This is a project that aims to predict the embryo stages in
its early development. The stages of embryo development is 
indicated by the number of blastomer of embryo. In this project
we detect up to 5-cell stage. This project utilize CRF that 
developed by Khan et, al [1]. This project using mouse embryo 
dataset from celltracking.bio.nyu.edu. This project has a good 
result in predicting 1-cell, 2-cells, and 4-cells stages, worst 
in predicting 3-cell stage, Acceptable for predicting 5-cells 
stages. 

Overall this project has an unique point in feature extraction.
we do not using common feature extraction as GLCM, SIFT, or Bag of 
Visual Words. In this project we use histogram of distance of edge 
to extract the feature from an embryo image.

It is very recomended running this project in python 3.5 environment. 

# Image Pre-Processing
we utilize skimage library to perform pre-processing. pre-processing
is the most important step in this project. there are two independent
pre-processing. the first one is used in frame based prediction.
and the other is used for transition prediction. 

the first one utilize frangi filter and sobel to do edge detection.
the purpose of this preprocessing is to extract embryo membrane from
mouse embryo image.

and the second utilize sobel only to do edge detection. the purpose 
of this preprocessing is to extract 'edge' shape from embryo image.

# Feature
from both preprocessing then all points that represent edge will be
extract and will be used to constructed list of points P={(x1,y1),(x2,y2),..
,(xn,yn)}. using a points called 'center points' then we will compute the distance
of all points in list of points P to produce a histogram of distance. this 
histogram of distance will be used as feature for frame based prediction and
transition prediction.

special for transition prediction, the feature will have more computation
to calculate the difference of feature from two contigous frames.

# Frame based Prediction
frame based prediction (fbp) aims to make a prediction based on one independent frame.
fbp will predict probabiity from each number of blastomer from one frame. in the process
we do feature selection for improve the prediction performance. we utilize binary bat algorithm [2]
to do the selection. we use 30 iteration and 20 particle (artificial bat) in utilizing bba.

# Transition Prediction
transition prediction aims to make a prediction of probability of transition from two 
contigous frames.

# CRF
Conditional random field is utilized to stabilize the prediction. combination of 
frame based prediction and transition prediction is used in the process. more frames
in sequance will produce better result in prediction.

# Experiment
we recomend do experiment with portion of train and test 50-50 percent. for time purpose
we only use 10 sequance from original dataset (train: E00,E06,E11,E16E25 || test:E03,E24,E26,E27,E30). 
we do experiment several times to get the best result. each experiment will produce different 
result based on which feature is selected.

run experiment : LS.py

# Visualization
we provide simple program to see performance of our model. run the script and you will see a simple GUI.
with this GUI you can choose the sequance folder and the model will make a prediction of all frame in 
this sequance. while the process is running please do not intervene the GUI.

run : GUIAEP.py

refrence:
[1] A. Khan, S. Gould, and M. Salzmann,“Automated monitoring of human embryonic cells up to the 5-cell
stage in time-lapse microscopy images,” 12th International Symposium
on Biomedical Imaging (ISBI), pp. 389–393, 2015.
[2] Mirjalili, S., Mirjalili, S. M., dan Yang, X.-S,"Binary bat algorithm," Neural
Computing and Applications, 25(3):663–681, 2014.

# Folder structure
if you want to add some sequance in any folder please follow these structure.

traing
|
--sequance1
  |
  --class1
    |
	--frame1
	|
	--frame2
	|
	|
	.
	.
	--framen
  |
  --class2
  |
  .
  .
  |
  --class5
|
--sequance2
.
.
|
--sequance m

test
|
--sequance1
  |
  --class1
    |
	--frame1
	|
	--frame2
	|
	|
	.
	.
	--framen
  |
  --class2
  |
  .
  .
  |
  --class5
|
--sequance2
.
.
|
--sequance m

sequance
|
--sequance1
  |
  --frame1
  |
  --frame2
  .
  .
  |
  --framen