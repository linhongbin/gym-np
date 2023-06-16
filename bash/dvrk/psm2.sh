source bash/user_var.sh
source $DVRK_PATH/devel/setup.bash
# echo $DVRK_PATH/devel/setup.bash
qlacloserelays
roslaunch dvrk_robot dvrk_arm_rviz.launch arm:=PSM2 config:=$DVRK_PATH/src/cisst-saw/sawIntuitiveResearchKit/share/cuhk-daVinci-2-0/console-PSM2.json