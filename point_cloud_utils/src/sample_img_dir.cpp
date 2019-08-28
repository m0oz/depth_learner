#include "sample_img_dir.h"

using namespace std;

NodeSampleImgdir::NodeSampleImgdir(const ros::NodeHandle &nh, const ros::NodeHandle &pnh) :
    nh_(nh), pnh_(pnh) {
  // Initialize Publisher
  rgb_pub_ = nh_.advertise<sensor_msgs::Image>("rgb_images", 1);
  depth_pub_ = nh_.advertise<sensor_msgs::Image>("depth_images_gt", 1);
  loadParameters();
}

void NodeSampleImgdir::readPublishImage(string img_path, const ros::Publisher &pub_, const string encoding) {
  //  // Read the image file
  cv::Mat image;
  if (encoding=="rgb8") {
    image = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  }
  else {
    image = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
  }
  // Check for failure
  if (image.empty()) {
    ROS_INFO("Could not open or find the image");
    return;
  }

  // Convert cv image to ros image message
  // Pusblish image messages
  std_msgs::Header header;
  header.stamp = ros::Time::now();
  header.frame_id = "/world";
  sensor_msgs::ImagePtr imgmsg = cv_bridge::CvImage(header, encoding, image).toImageMsg();

  pub_.publish(imgmsg);
}

void NodeSampleImgdir::listDirectoryContents(vector<string> &filelist_rgb,
                                             vector<string> &filelist_depth) {
  DIR *dir;
  struct dirent *ent;
  const char *path_rgb_dir = rgb_dir_.c_str();
  const char *path_depth_dir = depth_dir_.c_str();

  // RGB open directory
  if ((dir = opendir(path_rgb_dir)) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
        std::string filename(ent->d_name);
        if (filename.find(rgb_format_) != std::string::npos) {
          filelist_rgb.push_back(path_rgb_dir + filename);
        }
    }
  sort(filelist_rgb.begin(), filelist_rgb.end());
  } else {
    ROS_ERROR("Could not find specified depth directory.");
  }

  // DEPTH open directory
  if ((dir = opendir(path_depth_dir)) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
        std::string filename(ent->d_name);
        if (filename.find(depth_format_) != std::string::npos) {
          filelist_depth.push_back(path_depth_dir + filename);
        }
    }
  sort(filelist_depth.begin(), filelist_depth.end());
  } else {
    ROS_ERROR("Could not find specified depth directory.");
  }
}

void NodeSampleImgdir::loadParameters() {
  nh_.param<std::string>("/sample_img_dir/rgb_dir", rgb_dir_, "./rgb");
  nh_.param<std::string>("/sample_img_dir/depth_dir", depth_dir_, "./depth");
  nh_.param<std::string>("/sample_img_dir/rgb_format", rgb_format_, ".jpg");
  nh_.param<std::string>("/sample_img_dir/depth_format", depth_format_, ".png");
  nh_.param<float>("/sample_img_dir/sampling_rate", rate_, 1.0);
  ROS_INFO("Loaded param rgb_dir: %s", rgb_dir_.c_str());
  ROS_INFO("Loaded param depth_dir: %s", depth_dir_.c_str());
  ROS_INFO("Parsing for img formats: rgb (%s), depth (%s)",
            rgb_format_.c_str(), depth_format_.c_str());
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "sample_img_dir");
  ROS_INFO("Node Started");

  NodeSampleImgdir node_sampleImages;

  // Create list of files in image directory
  vector<string> filelist_rgb;      // list files with rgb file ext
  vector<string> filelist_depth;    // list files with depth file ext
  node_sampleImages.listDirectoryContents(filelist_rgb, filelist_depth);

  // Perform sanity check to ensure both lists have the same length
  if (filelist_rgb.size() != filelist_depth.size()) {
      ROS_ERROR("Length of rgb filelist does not match length of depth filelist");
  }

  // Set sampling rate in Hz
  ros::Rate rate(node_sampleImages.rate_);
  while(ros::ok()) {
    for (int i=0; i<filelist_rgb.size(); i++) {
      string rgb_path = filelist_rgb[i];
      string depth_path = filelist_depth[i];
      cout << rgb_path << " " << depth_path << endl;;
      node_sampleImages.readPublishImage(rgb_path, node_sampleImages.rgb_pub_, "rgb8");
      node_sampleImages.readPublishImage(depth_path, node_sampleImages.depth_pub_, "mono8");
      ROS_INFO("Published imgs");
      rate.sleep();
    }
  }

  return EXIT_SUCCESS;
}
