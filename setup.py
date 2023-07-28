from setuptools import setup, find_packages

setup(name='gym_np', 
      version='1.0',
    install_requires=[
        'gym', 
        'opencv-python==4.2.0.34',
        'tqdm',
        'simple_pid',
        'rospy',
        'rospkg',
        'spatialmath-python',
        'sympy',
        'roboticstoolbox-python',
        ], 
      packages=find_packages())
