# **gym-np**: Open-ai gym for needle picking to learn visuomotor policy

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
  cd gym-np
  git submodule init
  git submodule update
  ```

## Install

- Install [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) for supporting AMBF simulation

- Install [Anaconda](https://www.anaconda.com/download) for pythonic virtual environment

- Modified setting file, [setting.sh](./bash/setting.sh), if needed

- Install gym_np through bash files: :

  ```sh
  source ./bash/setting.sh # claim setting variables
  source ./bash/install/conda.sh # create conda environment
  source ./bash/install/ambf.sh # install AMBF, will take some time to compile
  source ./bash/install/other.sh # install other packages
  ```

## Launch simulator

  For all command lines, we assume the current directory is `<path to gym-np>`.

- **Launch roscore, open 1st terminal and type**
  ```
  roscore
  ```

- **Launch simulator, open 2st terminal and type**
  ```sh
  source ./bash/ambf/sim_standard.sh # For standard needle
  ```
  > For needle variations, type the following commands instead:
  >
  >  ```sh
  >  source ./bash/ambf/sim_small.sh # >For small needle
  >  ```
  >  ```sh
  >  source ./bash/ambf/sim_large.sh # For large needle
  >  ```
  >  ```sh
  >  source ./bash/ambf/sim_irregular1.sh # For irregular shape 1
  >  ```
  >  ```sh
  >  source ./bash/ambf/sim_irregular2.sh # For irregular shape 2
  >  ```

- **Launch crtk interface for control, open 3st terminal and type**
  ```sh
  source bash/ambf/crtk.sh
  ```

- **Reset robotic arm to ready position, open 4st terminal and type**
  ```sh
  source bash/ambf/reset.sh 
  ```

## Example

  A user-interactive example code can be run by:
  ```sh
  source bash/ambf/init.sh
  python example/env_play.py --action oracle
  ```