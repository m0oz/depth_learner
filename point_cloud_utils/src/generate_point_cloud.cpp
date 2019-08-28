#include "generate_point_cloud.h"
#include "point_cloud_utils/Keypoints.h"

using namespace message_filters;
using namespace std;

GeneratePointCloud::GeneratePointCloud(const ros::NodeHandle &nh,
                                       const ros::NodeHandle &pnh):
    nh_(nh), pnh_(pnh) {

  // Sync messages and release synced bundles of either rgb/depth or features/depth
  gt_depth_sub_ = new Subscriber<sensor_msgs::Image>(
                                              nh_, "depth_images_gt", 3);
  pred_depth_sub_ = new Subscriber<sensor_msgs::Image>(
                                              nh_, "depth_images_pred", 3);
  rgb_sub = new Subscriber<sensor_msgs::Image>(
                                              nh_, "rgb_images", 3);
  features_sub_ = new Subscriber<point_cloud_utils::Keypoints>(
                                              nh_, "features", 3);
  // Initiate synchronizer with approximate sync policy
  syncSLAM = new Synchronizer<slamSyncPolicy>(slamSyncPolicy(5),
                                            *gt_depth_sub_, *features_sub_);
  syncCNN = new Synchronizer<cnnSyncPolicy>(cnnSyncPolicy(5), *pred_depth_sub_,
                                            *gt_depth_sub_, *rgb_sub);
  // Assign callback functions for synchronizers
  gt_depth_sub_->registerCallback(&GeneratePointCloud::messageCallbackDenseGT, this);
  pred_depth_sub_->registerCallback(&GeneratePointCloud::messageCallbackDenseCNN, this);
  syncSLAM->registerCallback(boost::bind(
          &GeneratePointCloud::messageCallbackSparseSLAM, this, _1, _2));
  syncCNN->registerCallback(boost::bind(
          &GeneratePointCloud::messageCallbackSparseCNN, this, _1, _2, _3));

  // Init publishers
  pcl_dense_gt_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("pcl_dense", 1);
  pcl_cnn_dense_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("pcl_dense_cnn", 1);
  pcl_cnn_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("pcl_cnn", 1);
  pcl_cnn_gt_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("pcl_cnn_groundtruth", 1);
  pcl_slam_gt_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("pcl_slam_groundtruth", 1);

  // Load params from launch file
  loadParameters();

  // Initialize ORB feature extracotr
  orb_detector_ = cv::ORB::create(2000);

  counter_dense = 0;
  counter_cnn = 0;
  counter_slam = 0;
}

void GeneratePointCloud::generatePcl(const cv::Mat &image,
                                     vector<vector<int>> &features,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr pcl) {
  // Check for failure
  if (image.empty()) {
    std::cout << "Could not open image" << std::endl;
    return;
  }

  // Be sure that the image is 8bpp and single channel
  if (image.type() != CV_8U || image.channels() != 1) {
    // Wrong image depth or channels
    std::cout << "Wrong image depth or channels" << std::endl;
    return;
  }

  double fx = (image_cols_ / 2.0) / (std::tan((fov_h_ / 2.0) / 180.0 * M_PI));
  double fy = (image_rows_ / 2.0) / (std::tan((fov_v_ / 2.0) / 180.0 * M_PI));
  // assuming focal point is in center of image
  double cx = (image_cols_ / 2.0);
  double cy = (image_rows_ / 2.0);
  int cols = image.cols;
  int rows = image.rows;

  double unit_scaling = 0.1;
  for (vector<vector<int>>::iterator it=features.begin(); it!=features.end(); it++) {
    int x = (*it)[0];
    int y = (*it)[1];
    unsigned char Z_short = image.at<unsigned char>(y, x);
    auto Z = static_cast<double>(Z_short);
    if (Z != 0) {
      Z *= unit_scaling;
      pcl->push_back(pcl::PointXYZ(static_cast<float>(((x-cx)*Z)/fx),
                                   static_cast<float>(((y-cy)*Z)/fy),
                                   static_cast<float>(Z)));
    }
  }
  pcl->header.frame_id = "/world";
  ros::Time stamp = ros::Time::now();
  pcl->header.stamp = stamp.toNSec()/1e3;
}

void GeneratePointCloud::messageCallbackSparseSLAM(const sensor_msgs::ImageConstPtr& depthmsg,
                                             const point_cloud_utils::KeypointsConstPtr& featuremsg) {
  ROS_INFO("(%d) Received depth and features: generating SLAM pointcloud...", counter_slam);
  cv::Mat depth;
  try {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(depthmsg);
    depth = cv_ptr->image;
  } catch(cv::Exception& e) {
    const char* err_msg = e.what();
    std::cout << "exception caught when converting imgmsg: " << err_msg << std::endl;
  }

  vector<vector<int>> features;

  for (int i=0; i<featuremsg->numPoints; i++) {
    std_msgs::Int32MultiArray keypoint  = featuremsg->keypoints[i];
    features.push_back({keypoint.data[0], keypoint.data[1]});
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl(new pcl::PointCloud<pcl::PointXYZ>);
  generatePcl(depth, features, pcl);
  pcl_slam_gt_pub_.publish(pcl);
  counter_slam++;
}

void GeneratePointCloud::messageCallbackSparseCNN(const sensor_msgs::ImageConstPtr& preddepthmsg,
                                            const sensor_msgs::ImageConstPtr& gtdepthmsg,
                                            const sensor_msgs::ImageConstPtr& rgbmsg) {
  ROS_INFO("(%d) Received cnn-depth and rgb: gen cnn pointcloud", counter_cnn);
  cv::Mat depth_pred, depth_gt, rgb;
  try {
    cv_bridge::CvImageConstPtr cv_ptr_pred = cv_bridge::toCvCopy(preddepthmsg);
    cv_bridge::CvImageConstPtr cv_ptr_gt = cv_bridge::toCvCopy(gtdepthmsg);
    cv_bridge::CvImageConstPtr cv_ptr_rgb = cv_bridge::toCvCopy(rgbmsg);
    depth_gt = cv_ptr_gt->image;
    depth_pred = cv_ptr_pred->image;
    rgb = cv_ptr_rgb->image;
  } catch(cv::Exception& e) {
    const char* err_msg = e.what();
    std::cout << "exception caught when converting imgmsg: " << err_msg << std::endl;
  }

  vector<vector<int>> features = {};
  extractFeatures(rgb, features);

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_gt(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pred(new pcl::PointCloud<pcl::PointXYZ>);
  generatePcl(depth_gt, features, pcl_gt);
  generatePcl(depth_pred, features, pcl_pred);
  pcl_cnn_gt_pub_.publish(pcl_gt);
  pcl_cnn_pub_.publish(pcl_pred);
  counter_cnn++;
}

void GeneratePointCloud::messageCallbackDenseCNN(const sensor_msgs::ImageConstPtr& depthmsg) {
  cv::Mat depth;
  try {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(depthmsg);
    depth = cv_ptr->image;
  } catch(cv::Exception& e) {
    const char* err_msg = e.what();
    std::cout << "exception caught when converting imgmsg: " << err_msg << std::endl;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl(new pcl::PointCloud<pcl::PointXYZ>);
  vector<vector<int>> features;
  for (int i=0; i<depth.rows; ++i) {
    for (int j=0; j<depth.cols; ++j) {
      features.push_back({j, i});
    }
  }
  generatePcl(depth, features, pcl);
  pcl_cnn_dense_pub_.publish(pcl);
  counter_dense++;
}

void GeneratePointCloud::messageCallbackDenseGT(const sensor_msgs::ImageConstPtr& depthmsg) {
  cv::Mat depth;
  try {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(depthmsg);
    depth = cv_ptr->image;
  } catch(cv::Exception& e) {
    const char* err_msg = e.what();
    std::cout << "exception caught when converting imgmsg: " << err_msg << std::endl;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl(new pcl::PointCloud<pcl::PointXYZ>);
  vector<vector<int>> features;
  for (int i=0; i<depth.rows; ++i) {
    for (int j=0; j<depth.cols; ++j) {
      features.push_back({j, i});
    }
  }
  generatePcl(depth, features, pcl);
  pcl_dense_gt_pub_.publish(pcl);
  counter_dense++;
}

void GeneratePointCloud::extractFeatures(const cv::Mat &image,
                                         vector<vector<int>> &features) {
  vector<cv::KeyPoint> keypoints;
  orb_detector_->detect(image, keypoints);

  vector<cv::KeyPoint>::iterator it;
  for( it = keypoints.begin(); it != keypoints.end(); it++)
  {
    features.push_back({static_cast<int>(it->pt.x), static_cast<int>(it->pt.y)});
  }
}

void GeneratePointCloud::loadParameters() {
  nh_.param<double>("image_height", image_rows_, 240.0);
  nh_.param<double>("image_width", image_cols_, 320.0);
  nh_.param<double>("fov_v", fov_v_, 45.0);
  nh_.param<double>("fov_h", fov_h_, 60.0);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "pcl_generator");
  ROS_INFO("Node Started");

  GeneratePointCloud pcl_generator;

  ros::spin();

  return EXIT_SUCCESS;
}
