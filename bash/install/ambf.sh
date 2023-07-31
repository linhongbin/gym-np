source /opt/ros/noetic/setup.bash
sudo apt install libasound2-dev libgl1-mesa-dev xorg-dev
conda install -c conda-forge catkin_pkg -y
conda install -c anaconda cmake libffi -y
conda install -c aquaveo libp11 -y
pip install PyYAML empy
mkdir -p build/ambf
pushd build/ambf
source /opt/ros/noetic/setup.bash
cmake ../../ext/ambf
make -j4
popd