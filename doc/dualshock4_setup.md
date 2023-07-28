## Install

Since we use python3 in our virtual environment, dualshock is needed to be installed using python3 environment. We tested in both `noetic` and `melodic`.


Install `ds4drv` to virtual environment
```sh
cd 
git clone https://github.com/naoki-mizuno/ds4drv --branch devel
cd ds4drv/
cd <path-to-gym_np>
source bash/user_var.sh
python setup.py install --prefix $ANACONDA_PATH/envs/gym_np
sudo cp udev/50-ds4drv.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```


Downdload `ds4_driver` to ros workspace for gym_np
```sh
cd
mkdir -p gym_np_ws/src
cd gym_np_ws/src
git clone https://github.com/naoki-mizuno/ds4_driver.git -b noetic-devel # even for melodic user, you need to install noetic branch. Since melodic branch of ds4_driver is bugged with python2, and we need to sync with python3
```

compile workspace where python version is sync with gym_np virtual environment
```sh
cd
cd gym_np_ws
catkin_make -DPYTHON_EXECUTABLE=$ANACONDA_PATH/envs/gym_np/bin/python
```

For those encounter errors using `melodic`, install missing packages
```sh
source <path-to-gym_np>/bash/ambf/init.sh # just activate virtual environment
pip uninstall empy
pip install empy # install whatever packages when catkin_make complained about missing
```

## Test Install

```sh
cd <path-to-gym_np>
source bash/dvrk/init.sh
python
import ds4_driver # if no error, then install sucessfully
```