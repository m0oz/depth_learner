cmake_minimum_required(VERSION 2.8.3)
project(point_cloud_utils)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(OpenCV 3)
find_package(PCL REQUIRED)
find_package(catkin_simple REQUIRED COMPONENTS)

catkin_simple()

cs_add_executable(generate_point_cloud src/generate_point_cloud.cpp)
cs_add_executable(sample_img_dir src/sample_img_dir.cpp)
target_link_libraries(generate_point_cloud ${OpenCV_LIBRARIES})
target_link_libraries(sample_img_dir ${OpenCV_LIBRARIES})

cs_export()
