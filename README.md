# golfcart-workspace

Building OpenCV Bridge for Python3 is required. Follow instructions below:
```bash
sudo apt-get install python-catkin-tools python3-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-melodic-cv-bridge
cd catkin_ws
# Instruct catkin to set cmake variables
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
# Instruct catkin to install built packages into install place. It is $CATKIN_WORKSPACE/install folder
catkin config --install
# Clone cv_bridge src
git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv
# Find version of cv_bridge in your repository
apt-cache show ros-melodic-cv-bridge | grep Version
    Version: 1.13.0-0bionic.20191008.195111
# Checkout right version in git repo. In our case it is 1.13.0
cd src/vision_opencv/
git checkout 1.13.0
cd ../../
# Build
catkin build cv_bridge -DCATKIN_ENABLE_TESTING=0
# Build everything
catkin build -DCATKIN_ENABLE_TESTING=0
# Extend environment with new package
source install/setup.bash --extend
```

Taken from https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3

Then, to run visualization on ROS bag:
- launch `roscore`
- play rosbag file in a loop with `rosbag -l file.bag`
- launch visualization script with `rosrun visualization subscriber.py`
