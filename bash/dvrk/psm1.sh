source bash/user_var.sh
source $DVRK_PATH/devel/setup.bash
# echo $DVRK_PATH/devel/setup.bash
qlacloserelays
roslaunch dvrk_robot dvrk_arm_rviz.launch arm:=PSM1 config:=$DVRK_PATH/src/cisst-saw/sawIntuitiveResearchKit/share/cuhk-daVinci-2-0/console-PSM1.json