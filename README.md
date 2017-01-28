# CarND-Behavioral-Cloning-Project
The project is about cloning the driving behavior by training a convolutional neural network (CNN) to map raw pixels from a single front-facing center camera directly to steering commands. 

**Problem statement**: Train the CNN for track1 and test on same track by driving the car in simulator.The video is available
https://youtu.be/V5Ur3tfxDKg

**Self Assessment**: Test the CNN model for track 2, that demonstrates the generalization of the trained model.The video is available
https://youtu.be/WkKWzckqj90

##Pre-requisites
Windows 64 bit Beta simulator used

Udacity track 1 dataset


##Repository structure

**model.py** : 
           This python file has script to load the training dataset which has been resized to 32x16. The original images provided by
           Udacity as Track1 is of size 320x160. Data augmentation is done by flipping,brightness adjustment,recovery data generated from
           left and right camera images.Using keras with Tensorflow as backend, the model is created and trained with Track1 Udacity
           dataset. 
           
**drive.py** : 
           This python file is modified to include preprocessing of image(as same as done during training phase) fed to CNN for inference
           phase. For Track1 and Track2, different throttle values are set.For Track 1, throttle value set to 0.16. Before running for Track 2 , comment this.
           
By running model.py, the following files are generated which has saved model architecture and weights.

**model.json**: Has model architecture.

**model.h5**  : Has model weights. 

**Data_Visualization_Preparation.ipynb** : This file has visualization of 
            Initial dataset for analysis to know whether data is balanced or imbalanced.
            Visualization after each steps used in augmentation of dataset.Also visualization of internal CNN states
            
**training_dataset_resized.p**: This pickled file is generated which has resized images of 32x16 size.The 3 camera dataset is segregated
            and saved. The script is available in **Data_Visualization_Preparation.ipynb** file.The udacity data having images of
	    resolution 320x160 is read and resized.The section of code is under Markdown cell. Udacity data is not uploaded in
	    repository owing to large size.
	    
**png files**: These files are used for displaying visuals in this README.md file
	    

##Data Preprocessing
The Udacity track 1 dataset is used as training dataset.
The original images of training dataset are of resolution 320x160. In order to run training and inference phases on CPU, and also the DNN need to extract features(the road) only in high level, considered resizing to 32x16 and stored as pickled file.Data augmentation is carried out using below mentioned techniques. These augmented datset is converted from RGB to HSV color space and only Saturation channel is used for training the CNN.

![HSV](/image_HSV.png)

##Data Analysis and Augmentation
###Initial training set has following distribution
![Steers Hist](/hist_org.png)

straight_steers:4361        left_steers:1775     right_steers:1900       Total samples:8036

The approach used to solve this problem is End to End learning using DNN.Since prediction of steering angle is 
continuous , it is a regression problem. Eventhough left angle and right angle data are equally distributed,the straight angle data are almost 50% of total data. This may result in prediction biased towards straight steering. To remove the imbalance in data distribution in training set, data augmentaion is done with flipping center camera images, also the car to recover when gets off to the side of the road,left and right camera images are used with static offset  added/subtracted respectively to steer angles.Finally these inputs are brightness adjusted and augmented resulting in twice amount of training inputs.

###Flipping Center Camera images

![Flipped](/flipping_augment.png)

###Brightness Augmentation

![Brightness](/brightness_augment.png)

###After data augmentation the training set has following distribution

![Steers Aug Hist](/hist_augment.png)

straight_steers:17444        left_steers:23366     right_steers:23478       Total samples:64288
            
##Model Achitecture
The network architecture is simple ConvNet of about 373 parameters.
It has a normalization layer, 1 convolution2D layer,Activation is ELU,Maxpooling layer with dropout is used.
Output is steering control. 

The model is trained with following hyperparameters

Epochs: 10, Batch Size: 128, Adam optimizer is used, Loss function: MSE .The augmented training dataset is split into training and validation set as 80/20.The training data will be randomly shuffled at each epoch while validation set is not shuffled. 
						
	32x16x1 input ---> normalization layer ---> conv layer ---> ELU ----> Maxpooling ----> dropout------> output
						
 ----   **Normalization layer**:
						It normalizes on the fly the input 'S channel' to be within the range -0.5 to 0.5 as same as used in comma.ai amd nvidia
						models.
						
 ----   **Convolution layer**: 
						 Filter kernel size (3,3) and number of filters is 12. Input shape(16,32,1) and output shape(14,30,12)
						 
 ----   **Activation**: 
 							ELU (Exponential linear unit). Instead of ReLU, this improved in converging faster
							
 ----   **MaxPooling**: Input shape(14,30,12) and output shape(3,7,12) pool_size : (4, 4), strides: (4, 4)
 							provides a form of translation invariance and thus benefits generalization
							
 ----    **Dropout(0.25)**: Input shape(3,7,12),output shape(3,7,12)
 														Regularization.

 ----    **Flatten**:  Input shape(3,7,12), output shape(252)
 
 
 ----    **Dense(1)**  : input shape(252), output shape(1)
 
###Visualization of internal CNN state
 This demonstrates that the CNN learned to detect useful road features on its own,i.e., with only the human steering angle as training signal. We  never explicitly trained it to detect the outlines of roads. By training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor and which serve as controller.
(Reference end-to-end-dl-using-px.pdf nvidia whitepaper)
 
![Visual_conv_input](/conv_layer_input.png)

![Visual_conv_output](/conv_layer_output.png)

    						
###Summary

Initially started by adapting comma.ai and nvidia models. Comma.ai model didn't work out for me to solve this problem.
Explored with nvidia model helped in driving car in Track1.But failed to work in Track 2.My interpretation is for this number of training dataset, it could be due to overfitting.As discussion was happening in forum, this could be due to million parameter models overfit , so some of them were showing success with smaller model. As my requirement is to run on CPU as i don't have GPU, i started to work with smaller convnet models.And one more issue is simulator (windows 64 bit) was taking almost 100% CPU to run and inference with original size(320x160 or higher resolution) needs higher computational requirements,this will aggravate the performance of model in simulator.This is the reason to work upon low resolution images(32x16) and small convenet model.  

Data preparation played major role in getting CNN trained to tackle steep curves(by augmenting with flipping images),the car to recover from drifting off the road by augmenting with left and right camera images by adding/subtracting offset to steering angles and brightness augmentation helped in making CNN robust to varying light conditions.

The network is trained with trial and error by varying number of filters in Convolution layer and choosen 12, kernel size (3,3), shuffle training data after each epoch,Maxpooling strides,dropout probability etc and checked the performance on track 1 in simulator. Adjusted the throttle values in drive.py.

The simple convnet model trained for track 1 works well for track 2 also which means the model is trained for generalization.

I would like to thank few of forum members carnd.slack.com (my apologies for not naming individually as couldn't recollect their names) for providing insights in solving the problem. 
