source bash/user_var.sh
source $ANACONDA_PATH/bin/activate
conda activate $ENV_NAME
source $AMBF_BUILD_PATH/build/devel/setup.bash
export PYTHONPATH=$PYTHONPATH:$ANACONDA_PATH/envs/$ENV_NAME/lib/python3.7/site-packages:$ROS_WS_PATH/devel/lib/python3/dist-packages/
export LD_LIBRARY_PATH=$ANACONDA_PATH/envs/$ENV_NAME/lib/:/usr/local/lib/
