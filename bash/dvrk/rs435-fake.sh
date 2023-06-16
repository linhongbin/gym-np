source bash/user_var.sh
source $DVRK_PATH/devel/setup.bash
rosbag play ./gym_suture/model/rs435_fake_topic.bag -l
# json_file_path:=~/tes.json