[toc]

# CARLA-COSMOS-Learning by Cheating

This implementation is created to run imitation learning successfully in `COSMOS` map using `CARLA 0.9.13`. It is based on the original [repo](https://github.com/bradyz/2020_CARLA_challenge) and the paper cited below.

> [**Learning by Cheating**](https://arxiv.org/abs/1912.12294)    
> Dian Chen, Brady Zhou, Vladlen Koltun, Philipp Kr&auml;henb&uuml;hl,        
> [Conference on Robot Learning](https://www.robot-learning.org) (CoRL 2019)      
> _arXiv 1912.12294_

## Installation

### Environment Requirements

This code uses CARLA 0.9.13.

All python packages used are specified in `carla_project/requirements.txt`.

```bash
$ conda create --name lbc python=3.7
$ conda activate lbc
$ pip install -r requirements.txt
```

## Run a pretrained model

### Preparations and settings

One checkpoint is downloaded from their [Wandb project](https://app.wandb.ai/bradyz/2020_carla_challenge_lbc) named `model_epoch24.ckpt` saved in the root folder.

First remember to export your carla simulator path, change the line in `run.sh` file, 

```bash
export CARLA_ROOT=${PATH TO YOUR CARLA SIMULATOR ROOT}
```

and change `ROUTES` to the desire route path, here a sample cosmos path is used.

```bash
export ROUTES=results/cosmos_route_3.xml
```

### Run bash file

First, spin up a CARLA server

```bash
$ ./CarlaUE4.sh -world-port=2000 -resx=800 -resy=600 -opengl
```

then run the agent after changing the parameter settings in `run.sh` file.

```bash
$ ./run.sh
```

## Training models from scratch

### Dataset and Data Collection

They provide a dataset of over 70k samples collected over the 75 routes based on Carla built maps. However, as we would like to run the model on COSMOS map, it is better to re-collect and re-train the model. Instructions on the details could be found from the original dataset and https://github.com/dotchen/LearningByCheating.

### Commands for training from scratch

Run the stage 1 training of the privileged agent.

```shell
$ python3 -m carla_project.src.map_model --dataset_dir /path/to/data --hack
```

We use wandb for logging, so navigate to the generated experiment page to visualize training.

**Important**: If you're interested in tuning hyper-parameters, see `carla_project/src/map_model.py` for more detail.  

To see what hyper-parameters we used for our models, you can see all of them by navigating to the corresponding [wandb run config](https://wandb.ai/bradyz/2020_carla_challenge_lbc/runs/command_coefficient=0.01_sample_by=even_stage2/overview).

Training the sensorimotor agent (acts only on raw images) is similar, and can be done by

```shell
$ python3 -m carla_project.src.image_model --dataset_dir /path/to/data
```

## References

[1] https://github.com/dotchen/LearningByCheating

[2] https://github.com/bradyz/2020_CARLA_challenge

## Directory Organization

```bash
.
├── agents
│   ├── __init__.py
│   ├── navigation
│   │   ├── agent.py
│   │   ├── basic_agent.py
│   │   ├── behavior_agent.py
│   │   ├── controller.py
│   │   ├── global_route_planner_dao.py
│   │   ├── global_route_planner.py
│   │   ├── __init__.py
│   │   ├── local_planner_behavior.py
│   │   ├── local_planner.py
│   │   ├── roaming_agent.py
│   │   └── types_behavior.py
│   └── tools
│       ├── __init__.py
│       └── misc.py
├── carla_project
│   └── src
│       ├── carla_env.py
│       ├── collect_data.py
│       ├── common.py
│       ├── controller_model.py
│       ├── converter.py
│       ├── dataset.py
│       ├── dataset_wrapper.py
│       ├── image_model.py
│       ├── map_demo.py
│       ├── map_model.py
│       ├── models.py
│       ├── record.py
│       ├── replay.py
│       ├── scripts
│       │   ├── cluster_points.py
│       │   ├── cluster.py
│       │   └── weights.py
│       └── utils
│           └── heatmap.py
├── leaderboard
│   ├── data
│   │   ├── all_towns_traffic_scenarios_public.json
│   │   └── parse.py
│   ├── leaderboard
│   │   ├── autoagents
│   │   │   ├── agent_wrapper.py
│   │   │   ├── autonomous_agent.py
│   │   │   ├── dummy_agent.py
│   │   │   ├── human_agent_config.txt
│   │   │   ├── human_agent.py
│   │   │   ├── __init__.py
│   │   │   ├── npc_agent.py
│   │   │   └── ros_agent.py
│   │   ├── envs
│   │   │   ├── __init__.py
│   │   │   └── sensor_interface.py
│   │   ├── evaluator.py
│   │   ├── __init__.py
│   │   ├── scenarios
│   │   │   ├── background_activity.py
│   │   │   ├── __init__.py
│   │   │   ├── master_scenario.py
│   │   │   ├── route_scenario.py
│   │   │   ├── scenarioatomics
│   │   │   │   ├── atomic_criteria.py
│   │   │   │   └── __init__.py
│   │   │   └── scenario_manager.py
│   │   └── utils
│   │       ├── checkpoint_tools.py
│   │       ├── __init__.py
│   │       ├── result_writer.py
│   │       ├── route_indexer.py
│   │       ├── route_manipulation.py
│   │       ├── route_parser.py
│   │       └── statistics_manager.py
│   ├── README.md
│   ├── requirements.txt
│   ├── scripts
│   │   ├── code_check_and_formatting.sh
│   │   ├── Dockerfile.master
│   │   ├── make_docker.sh
│   │   ├── parse.py
│   │   ├── pretty_print_json.py
│   │   ├── run_evaluation.sh
│   │   └── set_new_scenarios.py
│   └── team_code
│       ├── auto_pilot.py
│       ├── base_agent.py
│       ├── image_agent.py
│       ├── map_agent.py
│       ├── pid_controller.py
│       ├── planner.py
│       ├── README.md
│       └── requirements.txt
├── model
│   ├── cosmos_route_3.txt
│   └── model_epoch24.ckpt
├── README.md
├── requirements.txt
├── results
│   ├── cosmos_route_1.txt
│   ├── cosmos_route_1.xml
│   ├── cosmos_route_2.txt
│   ├── cosmos_route_2.xml
│   ├── cosmos_route_3.txt
│   └── cosmos_route_3.xml
├── run_agent.sh
├── run.sh
└── scenario_runner
    ├── manual_control.py
    ├── metrics_manager.py
    ├── no_rendering_mode.py
    ├── README.md
    ├── requirements.txt
    ├── scenario_runner.py
    └── srunner
        ├── autoagents
        │   ├── agent_wrapper.py
        │   ├── autonomous_agent.py
        │   ├── dummy_agent.py
        │   ├── human_agent_config.txt
        │   ├── human_agent.py
        │   ├── __init__.py
        │   ├── npc_agent.py
        │   ├── ros_agent.py
        │   └── sensor_interface.py
        ├── data
        │   ├── all_towns_traffic_scenarios1_3_4_8.json
        │   ├── all_towns_traffic_scenarios1_3_4.json
        │   ├── all_towns_traffic_scenarios.json
        │   ├── no_scenarios.json
        │   ├── routes_debug.xml
        │   ├── routes_devtest.xml
        │   └── routes_training.xml
        ├── __init__.py
        ├── openscenario
        │   ├── 0.9.x
        │   │   ├── migration0_9_1to1_0.xslt
        │   │   ├── OpenSCENARIO_Catalog.xsd
        │   │   ├── OpenSCENARIO_TypeDefs.xsd
        │   │   └── OpenSCENARIO_v0.9.1.xsd
        │   └── OpenSCENARIO.xsd
        ├── scenarioconfigs
        │   ├── __init__.py
        │   ├── openscenario_configuration.py
        │   ├── route_scenario_configuration.py
        │   └── scenario_configuration.py
        ├── scenariomanager
        │   ├── actorcontrols
        │   │   ├── actor_control.py
        │   │   ├── basic_control.py
        │   │   ├── external_control.py
        │   │   ├── __init__.py
        │   │   ├── npc_vehicle_control.py
        │   │   ├── pedestrian_control.py
        │   │   ├── simple_vehicle_control.py
        │   │   └── vehicle_longitudinal_control.py
        │   ├── carla_data_provider.py
        │   ├── __init__.py
        │   ├── result_writer.py
        │   ├── scenarioatomics
        │   │   ├── atomic_behaviors.py
        │   │   ├── atomic_criteria.py
        │   │   ├── atomic_trigger_conditions.py
        │   │   └── __init__.py
        │   ├── scenario_manager.py
        │   ├── timer.py
        │   ├── traffic_events.py
        │   ├── watchdog.py
        │   └── weather_sim.py
        ├── scenarios
        │   ├── background_activity.py
        │   ├── basic_scenario.py
        │   ├── change_lane.py
        │   ├── control_loss.py
        │   ├── cut_in.py
        │   ├── follow_leading_vehicle.py
        │   ├── freeride.py
        │   ├── __init__.py
        │   ├── junction_crossing_route.py
        │   ├── maneuver_opposite_direction.py
        │   ├── master_scenario.py
        │   ├── no_signal_junction_crossing.py
        │   ├── object_crash_intersection.py
        │   ├── object_crash_vehicle.py
        │   ├── open_scenario.py
        │   ├── opposite_vehicle_taking_priority.py
        │   ├── other_leading_vehicle.py
        │   ├── route_scenario.py
        │   ├── signalized_junction_left_turn.py
        │   └── signalized_junction_right_turn.py
        ├── tools
        │   ├── __init__.py
        │   ├── openscenario_parser.py
        │   ├── py_trees_port.py
        │   ├── route_manipulation.py
        │   ├── route_parser.py
        │   ├── scenario_helper.py
        │   └── scenario_parser.py
        └── utilities
            └── code_check_and_formatting.sh

32 directories, 168 files
```
