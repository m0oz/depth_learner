#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <pcl_ros/point_cloud.h>
#include "point_cloud_utils/Keypoints.h"

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace message_filters;
using namespace std;

class GeneratePointCloud {
 public:
  GeneratePointCloud(const ros::NodeHandle &nh, const ros::NodeHandle &pnh);
  GeneratePointCloud() :
      GeneratePointCloud(ros::NodeHandle(), ros::NodeHandle("~")) {
  }
  ~GeneratePointCloud() = default;
  // Function to generate pcl from depth image and a set of img coords
  void generatePcl(const cv::Mat &image, vector<vector<int>> &features,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr pcl);
  void generatePcl2(const cv::Mat &image, pcl::PointCloud<pcl::PointXYZ>::Ptr pcl);
  // Callback functions
  void messageCallbackSparseSLAM(const sensor_msgs::ImageConstPtr& depthmsg,
                           const point_cloud_utils::KeypointsConstPtr& featuremsg);
  void messageCallbackSparseCNN(const sensor_msgs::ImageConstPtr& preddepthmsg,
                          const sensor_msgs::ImageConstPtr& gtdepthmsg,
                          const sensor_msgs::ImageConstPtr& rgbmsg);
  void messageCallbackDenseCNN(const sensor_msgs::ImageConstPtr& depthmsg);
  void messageCallbackDenseGT(const sensor_msgs::ImageConstPtr& depthmsg);
  // Utility functions
  void extractFeatures(const cv::Mat &image, vector<vector<int>> &features);
  void loadParameters();

 private:
  ros::NodeHandle nh_, pnh_;

  // Register different callback functions depending on operation mode
  typedef sync_policies::ApproximateTime
          <sensor_msgs::Image, point_cloud_utils::Keypoints> slamSyncPolicy;
  typedef sync_policies::ApproximateTime
          <sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> cnnSyncPolicy;
  Subscriber<sensor_msgs::Image> *gt_depth_sub_;
  Subscriber<sensor_msgs::Image> *pred_depth_sub_;
  Subscriber<sensor_msgs::Image> *rgb_sub;
  Subscriber<point_cloud_utils::Keypoints> *features_sub_;
  Synchronizer<slamSyncPolicy>* syncSLAM;
  Synchronizer<cnnSyncPolicy>* syncCNN;
  ros::Publisher pcl_dense_gt_pub_;
  ros::Publisher pcl_cnn_dense_pub_;
  ros::Publisher pcl_cnn_pub_;
  ros::Publisher pcl_cnn_gt_pub_;
  ros::Publisher pcl_slam_gt_pub_;
  double image_rows_;
  double image_cols_;
  double fov_v_;
  double fov_h_;
  int counter_dense, counter_cnn, counter_slam;
  cv::Ptr<cv::FeatureDetector> orb_detector_;
};
