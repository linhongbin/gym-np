# source ./bash/init.sh
source ./bash/setting.sh
# $AMBF_PATH/bin/lin-x86_64/ambf_simulator --launch_file ./launch.yaml -l 0,1,2,3,4,5,14,15 -p 200 -t 1 # full setup
$AMBF_PATH/bin/lin-x86_64/ambf_simulator --launch_file ./gym_np/asset/launch_modified.yaml -l 22,4,20 -p 200 -t 1 # just psm2 and simple stuff 
### warning!! the order of indexs can not be randomized