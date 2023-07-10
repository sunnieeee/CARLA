export CARLA_ROOT=/home/yuqing/CARLA/COSMOS_carla           # change to where you installed CARLA
export PORT=2000                                                    # change to port that CARLA is running on
export ROUTES=results/cosmos_route_3.xml         # change to desired route
export TEAM_AGENT=image_agent.py                                    # no need to change
export TEAM_CONFIG=model/model_epoch24.ckpt                                       # change path to checkpoint
export HAS_DISPLAY=1                                                # set to 0 if you don't want a debug window

./run_agent.sh
