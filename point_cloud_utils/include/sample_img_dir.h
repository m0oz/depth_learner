#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <dirent.h>

using namespace std;

class NodeSampleImgdir {
 public:
  NodeSampleImgdir(const ros::NodeHandle &nh, const ros::NodeHandle &pnh);
  NodeSampleImgdir() :
      NodeSampleImgdir(ros::NodeHandle(), ros::NodeHandle("~")) {
  }
  ~NodeSampleImgdir() = default;

  // Function definitions
  void loadParameters();
  void readPublishImage(string img_path_, const ros::Publisher &pub_, const string encoding);
  void listDirectoryContents(vector<string> &filelist_rgb, vector<string> &filelist_depth);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Publisher rgb_pub_;
  ros::Publisher depth_pub_;
  float rate_;

 private:
  std::string rgb_dir_;
  std::string depth_dir_;
  std::string rgb_format_;
  std::string depth_format_;
};
