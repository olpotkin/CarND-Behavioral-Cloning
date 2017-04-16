#**Behavioral Cloning** 

---

The goals / steps of this project are the following:

* Use the [simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior
* Build a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[image1]: ./images/001.PNG "Center lane driving"
[image2]: ./images/002.PNG "Input from three cameras"
[image3]: ./images/003.PNG "Flipped images"
[image4]: ./images/004.PNG "Cropped images"
[image5]: ./images/005.PNG "Model architecture"
[image6]: ./images/006.PNG "Visualization of the architecture"
[image7]: ./images/007.PNG "Recovery Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **video.py** for creating the video recording when car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **README.md** summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing:

```sh
python drive.py model.h5
```

In case of using Docker:

```sh
docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starter-kit python drive.py model.h5
```

Port 4567 is used by the simulator to communicate.

Setup of the development environment with the [CarND Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) is a required!

####3. Submission code is usable and readable

The **model.py** file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 178-196) 

The model includes RELU layers to introduce nonlinearity (lines 184-188), and the data is normalized in the model using a Keras lambda layer (code line 182). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py line 189).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 210-217). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 202).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving counter-clockwise on track one and track two.

For details about how I created the training data, see the next section. 


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to achieve excellent results during the drive in autonomous mode.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it shows good results in the task of [Traffic Sign Classification](https://github.com/olpotkin/CarND-Traffic-Sign-Classifier-Project) Project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implies that the model is overfitted. 

To combat the overfitting, I decided to use the [Nvidia-Autopilot](https://github.com/0bserver07/Nvidia-Autopilot-Keras) CNN architecture. This model is appropriate because it shows very good results in the [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I've added a dropout layer to the model in order to reduce overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There was a one spot where the vehicle fell off the track: road after the Bridge. To improve the driving behavior in these case, I performed recovering from the left and right sides of the road in problematic section.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

---

HERE IS THE [LINK ON YOUTUBE](https://www.youtube.com/watch?v=7-_1MnMiw1U) with example of driving in autonomous mode on the **TRACK ONE**.


####2. Final Model Architecture

The final model architecture (model.py lines 178-196) consisted of a convolutional neural network with the following layers and layer sizes:

![alt text][image5]

Here is a visualization of the architecture:

![alt text][image6]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn what to do when it is off on the side of the road.

Then I repeated this process on track two in order to get more data points.

To augment the data set, I used images from all three cameras with steering correction coefficient = 0.2:

![alt text][image2]

I also flipped images and angles thinking that this is an effective technique for helping with the left/right turn bias. For example, here is an image that has then been flipped:

![alt text][image3]

Then I cropped each image to focus on only the portion of the image that is useful for predicting a steering angle:

![alt text][image4]


After the collection process, I had 11123 (66738 with augmentation) number of data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The **ideal number of epochs was 6** as evidenced by the convergence of training loss and validation loss. 

![alt text][image7]

I used an **adam optimizer** so that manually training the learning rate wasn't necessary.

