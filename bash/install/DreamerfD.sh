conda install cudatoolkit=11.3 -c pytorch -y
pip install tensorflow==2.9.0 tensorflow_probability==0.17.0 pandas
conda install cudnn=8.2 -c anaconda -y
pip install protobuf==3.20.1
pushd ./ext/DreamerfD
pip install -e .
popd