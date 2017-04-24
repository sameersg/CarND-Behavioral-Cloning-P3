#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2017_04_06_22_01_09_367.jpg "Center Driving"
[image3]: ./examples/center_2017_04_06_22_02_25_168.jpg "Recovery Image"
[image4]: ./examples/center_2017_04_06_22_02_25_491.jpg "Recovery Image"
[image5]: ./examples/center_2017_04_06_22_02_25_628.jpg "Recovery Image"
[image6]: ./examples/center_2017_04_09_18_32_24_606.jpg "Recovery Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run1.mp4 video file

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on the NVIDIA Architecture its consists of 5 convolution layers using RELU as activation and 4 Dense layers. (model.py lines 73-82) 

The data is normalized in the model using a Keras lambda layer and cropping the images. (code line 70-71). 

####2. Attempts to reduce overfitting in the model

I had a Dropout layer, but the model was not overfitting with the Nvidia model and thats why i removed the dropout layer. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 29-64). The Dataset contains serveral normal and reverse driving and also normal driving in Track 2.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 84).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, reverse driving and normal driving on the second track. 
For training all three Kamera were used and also the augumented images of them. (model.py line 29-46 and 56-60)

###Model Architecture and Training Strategy

####1. Solution Design Approach


My first step was to use a convolution neural network model similar to the LeNet that i learned in the class. It was not working well for me, then i tried the Nvidia model and it was working much better with my Dataset, so i stick with it a started collecting Data to tune it. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 80% and 20%. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like the curve before and after the bridge. To improve that driving behavior in these cases, i collected extra data on how to recovering from the left and right sides in normal and reverse driving. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 68-82) is the Nvidia Model. 


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to stick at the center. These images show what a recovery looks like :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.
![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would give it a more generalized data set. And i also cropped the images so that it can focus on the neccessary parts of the image.

After the collection process i randomlly shuffled the data, i had 55617 training samples and 13905 validation samples.


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
