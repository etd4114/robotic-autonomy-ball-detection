This repo has a python file and a folder of images from the test video. For the project we can just copy the opencv flow to work with ROS.
## Quick Start
```bash
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone https://github.com/etd4114/robotic-autonomy-ball-detection.git
cd ..
catkin_make

# In one ternimal
source devel/setup.bash # if use zsh, source devel/setup.zsh
roslaunch robotic-autonomy-ball-detection detection.launch

# In another terminal
source devel/setup.bash # if use zsh, source devel/setup.zsh
rosbag play RA_ball_detection.bag

# open rviz and select /detections_results to see the result.
```
