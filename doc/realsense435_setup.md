## Install

1. remove all realsense
    ```sh
    dpkg -l | grep "realsense" | cut -d " " -f 3 | xargs sudo dpkg --purge
    ```
2. install librealsense2, following [link](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
    ```sh
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
    sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
    sudo apt-get install librealsense2-dkms
    sudo apt-get install librealsense2-utils
    ``` 
3. check driver by 
    ```sh
    realsense-viewer
    ```
4. check if firmware version is consistent with driver
    ```sh
    dpkg -l|grep realsense # driver version
    rs-fw-update -l # firmware version
    ```
    check this [website](https://dev.intelrealsense.com/docs/firmware-releases)
5. install ros wrapper 
    ```sh
    sudo apt-get install ros-$ROS_DISTRO-realsense2-camera
    sudo apt-get install ros-$ROS_DISTRO-realsense2-description
    ```

    check with rviz
    ```sh
    roslaunch realsense2_camera rs_camera.launch filters:=pointcloud
    ```


## Tune Realsense435 parameters

    ```sh
    realsense-viewer
    ```

https://blog.csdn.net/ahelloyou/article/details/115513850

## reference
https://github.com/IntelRealSense/realsense-ros/issues/2386#issuecomment-1264499208