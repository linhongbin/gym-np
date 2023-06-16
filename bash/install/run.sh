source ./bash/setting.sh
source $ANACONDA_PATH/bin/activate
conda create -n $ENV_NAME python=3.7 -y
conda activate $ENV_NAME



# install ambf
source ./bash/install/ambf.sh


# install python packages
source ./bash/install/other.sh


# install DreamerBC 
source ./bash/install/dreamerbc.sh