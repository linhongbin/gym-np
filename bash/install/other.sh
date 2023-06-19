conda install -c conda-forge python-orocos-kdl -y # conda install PyKDL using pre-compiled package

pushd ./ext/surgical_robotics_challenge/scripts/
pip install -e .
popd

# pushd ./ext/SurRoL
# pip install -e .
# echo "import surrol.gym" >> $ANACONDA_PATH/envs/$ENV_NAME/lib/python3.7/site-packages/gym/envs/__init__.py
# popd

pip install -e .