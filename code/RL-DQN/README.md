[toc]

# CARLA-COSMOS-Reinforcement-Learning

## Installation

### Environment Requirements

```shell
$ conda create --name dqn python=3.7
$ conda activate dqn
$ pip install -r requirements.txt
```

### Train DQN

First, sign up a CARLA server

```bash
$ ./CarlaUE4.sh -world-port=2000 -resx=800 -resy=600
```

Then, run the training script,

```bash
$ python src/dqn_main_agent.py
```

The trained model would be saved as `RL_model.ckpt` in the path indicated from the `dqn_params.py` file.

### Test trained model

```shell
$ python src/dqn_main_agent.py --test True
```

This command could not be used if there does not exit any saved models in `model` folder.

## Deep Q Network (DQN)

1. __State-space__ : The state is simply the 84 x 84 RGB image captured by the on-board vehicle camera which is processed by the neural network.
2. __Action-space__ : It is an array of tuples of the form (throttle, steering, brake). The output of the neural network is mapped to one of the tuples in this array using the minimum of the Euclidean norm calculated with respect to the obtained value from the neural network.
3. __Reward__ : The reward is computed in real-time using the wheel odometry, collision, lane-invasion sensor values and is discounted over an episode with a preset discounted factor.
4. __Neural Network Architecture__: 84 x 84 RGB image -> Convolution_1-> Convolution_2 -> Convolution_3 -> Flatten -> Fully Connected.
5. The output of the __Flatten layer__ is divided into the next __Fully Connected Layer__ and an Advantage output.
6. __Final Layer__: Q(s,a) = V(s) + A(s,a) - 1/|A| * sum(A(s,a'))

## Python File Descriptions

1. **`dqn_main_agent.py`**: This Python file is the central one which calls all the functions required for he CARLA simulator to run and the training and testing of the agent.  
2. **`dqn_utils.py`**: This Python file contains the classes - Sensors, DQNetwork, Memory and Sumtree. These are required for creating, modifying and adding/deleting the memory instances along with the Deep Neural Network.
   1. **`dqn_functions_extra.py`**: This Python file contains some auxiliary functions for the training and other operations. For the sake of easy access and reduced complexity and length of the code, I have collected all the functions here.

3. **`dqn_params.py`**: This Python file contains the hyper-parameters related to the memory buffer, neural network and the reinforcement learning agent.
4. **`manual_control.py`**: This Python file contains CARLA built class for applying control that need to be used in this application.

## Directory Organization

```{bash}
.
├── README.md
├── requirements.txt
└── src
    ├── dqn_functions_extra.py
    ├── dqn_main_agent.py
    ├── dqn_params.py
    ├── dqn_utils.py
    └── manual_control.py

1 directory, 7 files
```

## Reference

[1] https://github.com/erdos-project/pylot

[2] https://github.com/yang1688899/CarND-Vehicle-Detection

[3] https://github.com/Keshav0205/Deep-Reinforcement-Learning-for-Autonomous-Driving

[4] https://github.com/koustavagoswami/Autonomous-Car-Driving-using-Deep-Reinforcement-Learning-in-Carla-Simulator



