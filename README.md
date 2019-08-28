# Point clouds from CNN depth regression vs. SLAM

<figure>
 <img src="https://github.com/m0oz/depth_learner/blob/master/figures/uvd-DepthPredictions-10.jpg" width="500"/>
 <figcaption>
 Results images - Input RGB on the left, groundtruth depth and predictions on the right.
 </figcaption>
</figure>

This repository contains code for three different tasks
### 1. Trainable CNN for monocular depth prediction
A tensorflow model to predict depth maps from single RGB images.
The network architecture is depicted below. As reference we also implemented the model as proposed by [Eigen](https://arxiv.org/abs/1406.2283).
Compatible datasets:
* [NYU](https://cs.nyu.edu/~silberman/datasets/) (2.4k frames, RGBD, Kinect, idoor scenery, single frames)
* [SUNRGBD](http://rgbd.cs.princeton.edu/) (10k framess, RGBD, Kinect, indoor, single frames)
* [UVD](http://www.sira.diei.unipg.it/supplementary/ral2016/extra.html) (45k frames, RGBD, synthetic, indoor (city scape), 20 video sequences)
* [Unreal](https://isar.unipg.it/index.php?option=com_content&view=article&id=53:unrealdataset&catid=17&Itemid=212) (100k frames, RGBD, synthetic, outdoor (forest and city scape), video sequences)
* [SceneNet](https://robotvault.bitbucket.io/scenenet-rgbd.html) (5M frames, RGBD, synthetic, indoor, video sequence)

<figure>
 <img src="https://github.com/m0oz/depth_learner/blob/master/figures/resnet50_architecture.jpg" width="500"/>
 <figcaption>
 CNN architecture
 </figcaption>
</figure>

### 2. ROS package for online inference of depth maps
ROS node to load tensorflow model from checkpoint file and publish depth predictions for rgb images subsribed from a ROS topic.
### 3. ROS package to sample images, generate point clouds from depth images and evaluate point cloud quality
ROS node to generates and publishes dense or sparse point clouds depending on the inputs it receives:

| Received Messages  | Operation Mode (Output) |
| ------------------ | ----------------------- |
| **GT depth image** | Produce dense point cloud |
| **CNN depth prediction, RGB image, GT depth image** | Generate sparse point cloud from GT depth and ORB features extraced by an openCV ORB detector |
| **SLAM ORB features, GT depth image**  | Generate sparse point cloud from GT depth and slam ORB features  |

## Installation and Dependencies
How to setup everything step-by-step:
1. Clone this ```git clone https://github.com/uzh-rpg/semester_thesis_zimmermann.git depth-prediction```
2. (Optional if you want to run the ROS pipeline for online inference): 
   - Move or link the repo to the src directory of your ros_workspace
   - Make sure you have (catkin_simple)[https://github.com/catkin/catkin_simple] and (catkin_tools)[http://catkin-tools.readthedocs.io/en/latest/installing.html] installed(e.g. you are able to invoke ```catkin build```)
   - Make sure that PCL, Eigen3 and OpenCV are installed (PCL and OpenCV should be included in your ROS distribution)
   - Build your ros_workspace with ```catkin build online_inference```
2. Navigate to the root of this repo and edit ```setup.env.sh``` to update the paths to your ROS distribution and to your catkin workspace
3. Run ```. setup_env.sh```

Dependencies:
* Python (2.xx)
  * All required python packages are contained in requierements.txt and are installed using ```. setup_env.sh```
* ROS (only for point cloud evaluation)
  * [PCL](http://www.pointclouds.org/downloads/linux.html), [pcl_ros](http://wiki.ros.org/pcl_ros)
  * [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)
  * [OpenCV](https://opencv.org/), [cv_bridge](http://wiki.ros.org/cv_bridge)
  * [catkin simple](https://github.com/catkin/catkin_simple)
  
## Usage
If you do not intend to retrain the model you can download a pretrained model for the UVD dataset by invoking\
```
$ chmod +x download_model_uvd.sh
$ ./download_model_uvd.sh
``````
To get the actual dataset and prepare train and tests split as we used for our work use. (the dataset is about 10GB so it will run some minutes)
```
$ chmod +x download_data_uvd.sh$ 
$ ./download_data_uvd.sh
```
### Run ROS depth inference
* Make sure you have built the package "online_inference"
* Activate your virtualenv by calling ```. setup_env.sh```
* Only if your run this the first time: Update the launch with the paths to your image data and the path to your tensorflow checkpoint. If you used download_dat_uvd.sh to download you dont have to change anything.
  ```
  rosed online_inference experiment.launch
  ```
* Launch depth inference (possibly set a GPU as visible to speed up inference)
  ```
  roslaunch online_inference experiment.launch
  ```
* Launch rviz in another shell to visualize the produced point clouds
  ```
  roslaunch online_inference rviz.launch
  ```
### Visualize CNN depth predictions in a jupyter notebook
* Activate your virtualenv by calling ```. setup_env.sh```
* Navigate to ```cd depth_learner/src_depth_learner/```
 Launch jupyter and open visualie_predictions.ipynb
* If you want to use other data or models that the default ones adapt the config
### Train CNN for depth prediction
* Activate your virtualenv by calling ```. setup_env.sh```
* Navigate to ```cd depth_learner/src_depth_learner/```
* Possibly adapt config/resnet.yaml or models/nets.py to your needs (e.g. with your train and val directories)
* Start training and specify config and experiment name
  ```
  python train.py --config="configs/<your_config.yaml> --name="<your_exp_name>"
  ```
* Monitor training (your checkpoints will be stored under ./experiments/<your_exp_name>)
  ```
  tensorboard --logdir="experiments"
  ```
