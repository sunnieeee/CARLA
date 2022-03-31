# carla-vehicle-follow
## Requirements
We require that you run Python 3.7 and then can download all other required libraries by running the following.
```
pip install -r requirements.txt
```

Please note that to download the weights for yolov3 it will require GitHub Large File Storage refer here [Github Large File Storage](https://git-lfs.github.com/) for a download link and instructions on how to use it

## Autonomous Driving 
The main file to be run is `vehicle_test_NN.py` which is run like any other carla program. First the carla environment must be started up then the program is run as follows
```
python vehicle_test_NN.py
```
## Gathering Result
Video of the driving can be gathered by running the `one_folder.py` script. This script will compile all images genererated via inference into the `images/` folder thus allowing a user to see what the vehicle sees during its run. 

Please note that you must create this 'images/' folder within the directory yourself

Please note that you must either manually delete the `runs/` folder after every run of `vehicle_test_NN.py` or run `sudo python one_folder.py` which will give the program the ability to delete that folder for you.
