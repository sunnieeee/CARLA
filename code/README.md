[toc]

# DitecT Spring22 project -- CARLA

> Author: Yuqing Cao
>
> Dated: May 18th, 2022

Detailed introductions could be found at the `README.md` file within the corresponding folders. Here provides some brief descriptions and quick-start scripts to reproduce relevant results under this `code` root.

And before each execution, first make sure to run CARLA under the simulator's root directory.

```bash
./CarlaUE4.sh -world-port=2000 -resx=800 -resy=600
```

## Modified-Previous-Code

### Description

The previous code used yolo-v5 as the object detection network to collect camera sensor data and return controls to the vehicle. 

As a modified version, firstly, some bugs such as CARLA world reloading error, `runs` directory removing error, `images` directory not found error have been fixed. 

Additionally, a `img2vid.py` file has been created to realize the conversion from images to video to achieve a better visualization performance. 

Furthermore, the original files are sorted into `src` and `model` directories for a clearer organization.

### Execution

Create required environment.

```bash
conda create --name pre python=3.7
conda activate pre
cd Modified-Previous-Code
pip install -r requirements.txt
```

Run yolo-v5.

```bash
python src/vehicle_test_NN.py
```

Put all images into one folder and convert it into one video.

```bash
python src/one_folder.py
```

```bash
python src/img2vid.py
```

## RL-DQN

### Description

A reinforcement learning method, specifically, Deep Q Network, is implemented for autonomous driving in COSMOS map with CARLA simulation. 

### Execution

Create required environment.

```bash
conda create --name dqn python=3.7
conda activate dqn
cd RL-DQN
pip install -r requirements.txt
```

Run DQN training.

```bash
python src/dqn_main_agent.py
```

## Imitation-Learning

### Description

An imitation learning method, specifically, based on the paper Learning by Cheating, is implemented for autonomous driving in COSMOS map with CARLA simulation.

### Execution

Create required environment.

```bash
conda create --name lbc python=3.7
conda activate lbc
cd Imitation-Learning
pip install -r requirements.txt
```

Run the pretrained model.

```bash
# change to where you installed CARLA
export CARLA_ROOT=${PATH TO YOUR CARLA SIMULATOR ROOT}
export PORT=2000
export ROUTES=results/cosmos_route_3.xml
export TEAM_AGENT=image_agent.py
export TEAM_CONFIG=model/model_epoch24.ckpt
export HAS_DISPLAY=1   
```

```bash
./run_agent.sh
```

## Organization of the directory

```bash
./
├── Imitation-Learning
│   ├── agents
│   │   ├── __init__.py
│   │   ├── navigation
│   │   └── tools
│   ├── carla_project
│   │   └── src
│   ├── leaderboard
│   │   ├── data
│   │   ├── leaderboard
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── scripts
│   │   └── team_code
│   ├── model
│   │   └── model_epoch24.ckpt
│   ├── README.md
│   ├── requirements.txt
│   ├── results
│   │   ├── cosmos_route_1.txt
│   │   ├── cosmos_route_1.xml
│   │   ├── cosmos_route_2.txt
│   │   ├── cosmos_route_2.xml
│   │   ├── cosmos_route_3.txt
│   │   └── cosmos_route_3.xml
│   ├── run_agent.sh
│   ├── run.sh
│   └── scenario_runner
│       ├── manual_control.py
│       ├── metrics_manager.py
│       ├── no_rendering_mode.py
│       ├── README.md
│       ├── requirements.txt
│       ├── scenario_runner.py
│       └── srunner
├── Modified-Previous-Code
│   ├── model
│   │   ├── coco.names.txt
│   │   ├── steer_augmentation.h5
│   │   ├── yolov3.cfg
│   │   ├── yolov3-tiny.cfg
│   │   ├── yolov3-tiny.weights
│   │   ├── yolov3.weights
│   │   └── yolov5s.pt
│   ├── README.md
│   ├── requirements.txt
│   └── src
│       ├── img2vid.py
│       ├── one_folder.py
│       ├── vehicle_test_NN.py
│       └── yolov5s.pt
├── README.md
└── RL-DQN
    ├── README.md
    ├── requirements.txt
    └── src
        ├── dqn_functions_extra.py
        ├── dqn_main_agent.py
        ├── dqn_params.py
        ├── dqn_utils.py
        └── manual_control.py

20 directories, 41 files
```

