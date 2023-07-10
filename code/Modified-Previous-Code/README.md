[toc]

# carla-vehicle-follow

## Requirements
We require that you run Python 3.7 and then can download all other required libraries by running the following.
```shell
pip install -r requirements.txt
```

Please note that to download the weights for yolov3 it will require GitHub Large File Storage refer here [Github Large File Storage](https://git-lfs.github.com/) for a download link and instructions on how to use it

## Autonomous Driving 
The main file to be run is `vehicle_test_NN.py` which is run like any other carla program. First the carla environment must be started up then the program is run as follows
```shell
python src/vehicle_test_NN.py
```
## Gathering Result

### Move output images to one folder

Video of the driving can be gathered by running the `one_folder.py` script in the `src` folder. This script will compile all images genererated via inference into the `images/` folder thus allowing a user to see what the vehicle sees during its run. 

Please note that you must create this 'images/' folder within the directory yourself

Please note that you must either manually delete the `runs/` folder after every run of `vehicle_test_NN.py` or run `sudo python src/one_folder.py` which will give the program the ability to delete that folder for you.

### Convert image sequences to a video

```shell
python src/img2vid.py
```

The output video would be named as 'video.mp4' and saved in the root folder. If you would like to change the name or format of the video, please refer to the img2vid.py file.

## Directory Organization

```bash
.
├── model
│   ├── coco.names.txt
│   ├── steer_augmentation.h5
│   ├── yolov3.cfg
│   ├── yolov3-tiny.cfg
│   ├── yolov3-tiny.weights
│   └── yolov3.weights
├── README.md
├── requirements.txt
└── src
    ├── img2vid.py
    ├── one_folder.py
    └── vehicle_test_NN.py

2 directories, 11 files
```

