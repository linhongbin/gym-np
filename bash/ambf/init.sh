source bash/setting.sh
source $ANACONDA_PATH/bin/activate
conda activate $ENV_NAME
source $AMBF_BUILD_PATH/devel/setup.bash
export PYTHONPATH=$ANACONDA_PATH/envs/$ENV_NAME/lib/python3.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$ANACONDA_PATH/envs/$ENV_NAME/lib/:/usr/local/lib/
