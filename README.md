# CarND-Behavioral-Cloning-Project
The project is about cloning the driving behavior by training a convolutional neural network (CNN) to map raw pixels from a single front-facing center camera directly to steering commands. 

The repository has following folders and files.

**model.py** : 
           This python file has script to load the training dataset which has been resized to 32x16. The original images provided by
           Udacity as Track1 is of size 320x160. Data augmentation is done by flipping,brightness adjustment,recovery data generated from
           left and right camera images.Using keras with Tensorflow as backend, the model is created and trained with Track1 Udacity
           dataset. 
           
**drive.py** : 
           This python file is modified to include preprocessing of image(as same as done during training phase) fed to CNN for inference
           phase. For Track1 and Track2, different throttle values are set.
           
By running model.py, the following files are generated which has saved model architecture and weights.

**model.json**: Has model architecture.

**model.h5**  : Has model weights. 

**Data_Visualization_Preparation.ipynb** : This file has visualization of 
            Initial dataset for analysis to know whether data is balanced or imbalanced.
            Visualization after each steps used in augmentation of dataset.
            
**training_dataset_resized.p**: This pickled file is generated which has resized images of 32x16 size.The 3 camera dataset is segregated
            and saved. The script is available in **Data_Visualization_Preparation.ipynb** file.
            
 **Model Achitecture**:
 						The network architecture is simple ConvNet of about 373 parameters.
						It has a normalization layer, 1 convolution2D layer,Activation is ELU,Maxpooling layer with dropout is used.
						Output is steering control.
						
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
    							
