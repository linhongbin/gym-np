# **gym-np**: open-ai gym for needle picking to learn visuomotor policy

We try to provide a standard gym protocol for needle picking to learn a visuomotor policy using advanced RL algorithms. We built our code based on a the [2021-2022 AccelNet Surgical Challenge](https://collaborative-robotics.github.io/surgical-robotics-challenge/challenge-2021.html). For more information, please visit:
- [Project Website](https://sites.google.com/view/DreamerfD/home)
## Feature

  - Benchmark on [AccelNet Surgical Challenge](https://github.com/collaborative-robotics/surgical_robotics_challenge)
  - Support multiple needle variations
  - Integration with [gym](https://github.com/openai/gym)
## Dependency
  Since [AccelNet Surgical Challenge](https://github.com/collaborative-robotics/surgical_robotics_challenge) uses [AMBF](https://github.com/WPI-AIM/ambf) for simulation, it need to install [ROS](https://www.ros.org/) for its communications. We tested on 
 
  - OS Dependency: Ubuntu 20.04 
  
  - Software Dependency: [ROS Noetic](http://wiki.ros.org/noetic). [Anaconda](https://www.anaconda.com/download)

## Download 
  ```sh
  git clone https://github.com/linhongbin/gym-np.git
  git submodule init
  git submodule update
  ```

## Install

- Install [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) for supporting AMBF simulation

- Install [Anaconda](https://www.anaconda.com/download) for pythonic virtual environment

- Modified setting file, [setting.sh](./bash/setting.sh), if needed

- Install gym_np bash file: [run.sh](./bash/install/setting.sh), run the following command:

  ```sh
  source ./bash/install/run.sh
  ```
