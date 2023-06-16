source bash/user_var.sh
source $DVRK_PATH/devel/setup.bash
roslaunch realsense2_camera rs_camera.launch filters:=spatial,temporal,decimation,hole_filling align_depth:=true 
# disparity
# json_file_path:=~/tes.json