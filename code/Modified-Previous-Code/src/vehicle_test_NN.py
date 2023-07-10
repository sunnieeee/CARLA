import os
import sys
import numpy as np
import cv2
import time
import keras
import torch
import queue
torch.cuda.empty_cache()
from keras.models import load_model
from PIL import Image
import carla
import logging
from numpy import random
import time

IM_WIDTH = 640
IM_HEIGHT = 480

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
yolo_model.classes = [2] # limits this to just car detection

images = []

def main():

    actor_list = []
    all_id = []
    random.seed(int(time.time()))
    synchronous_master = False

    last_time = time.time()
    time.sleep(1)

    dirname = os.path.dirname(__file__)
    model_name = os.path.join(dirname,'../model/steer_augmentation.h5')
    model = load_model(model_name)

    '''
    Alternative YOLOv3 implementation
    # Load Yolo
    #yolo_model = cv2.dnn.readNet(os.path.join(dirname,"../model/yolov3-tiny.weights"), "yolov3-tiny.cfg")
    #yolo_model.classes = [2]
    #with open(os.path.join(dirname,"../model/coco.names.txt"), "r") as f:
    #    classes = [line.strip() for line in f.readlines()]
    #layer_names = yolo_model.getLayerNames()
    '''

    confidence_threshold = 0.6
    
    paused = False

    try:

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)


        # Once we have a client we can retrieve the world that is currently
        # running.
        world = client.get_world()
        new_settings = world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = 0.05
        world.apply_settings(new_settings) 
        #client.reload_world(False) # reload map keeping the world settings


        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        bp = world.get_blueprint_library().filter('model3')[0]

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        transform = random.choice(world.get_map().get_spawn_points())        

        # So let's tell the world to spawn the vehicle.
        #vehicle = world.spawn_actor(bp, transform)

        #generate waypoints
        waypoints = world.get_map().generate_waypoints(distance=1.0)
        print('Got Waypoints...')

        # Let's put the vehicle to drive around.
        #vehicle.set_autopilot(True)
        vehicle_blueprint = world.get_blueprint_library().filter('model3')[0]
        filtered_waypoints = []
        for waypoint in waypoints:
            if(waypoint.road_id == 10):
                filtered_waypoints.append(waypoint)

        spawn_point = filtered_waypoints[60].transform # 60
        spawn_point.location.z += 1
        spawn_point_lead = filtered_waypoints[120].transform # 120
        spawn_point_lead.location.z += 1

        vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
        vehicle_lead = world.spawn_actor(bp, spawn_point_lead)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov', '110')
        camera_bp.set_attribute('sensor_tick', '0.0')
        camera_bp.set_attribute('shutter_speed', '25.0')

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)
        
        actor_list.append(vehicle)
        actor_list.append(vehicle_lead)
        actor_list.append(camera)
        print('created %s' % vehicle.type_id)
        print('created %s' % vehicle_lead.type_id)

        def cameraListener(image):
            # Images
            img = np.array(image.raw_data)  # convert to an array
            img = img.reshape((IM_HEIGHT, IM_WIDTH, 4))  # was flattened, so we're going to shape it.
            img = img[:, :, :3]  # remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB)
            height, width, channels = img.shape

            screen = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))

            prediction = model.predict([screen.reshape(-1,160,120,1)])[0]
            print (prediction)
            steering_angle = prediction [0]
            
            throttle = prediction [1]
            brake = 0
            if throttle>=0:
                throttle = throttle/2
            else:
                throttle = 0


            print('Prelim throttle = ', throttle)
            print('Prelim steering = ', steering_angle)

            # Inference
            results = yolo_model(img)
            images.append(results)
            df = results.pandas().xyxy[0]
            
            # Custom transform for automated driving
            if not df.empty:
                closest_car = df.loc[0]
                w = closest_car['xmax']-closest_car['xmin']
                h = closest_car['ymax']-closest_car['ymin']
                x = closest_car['xmin']
                y = closest_car['ymin']
                diag_len = np.sqrt((w*w)+(h*h))
                print ('Diag Length = ', diag_len)
                if diag_len >= 150: # Detected object is too close so STOP
                    print("STOP")
                    throttle = 0
                    steering_angle = 0
                    brake = 1
                elif diag_len>=50 and diag_len<200: # Detected object is near by so SLOW DOWN
                    print("SLOW")
                    brake = 0.25
                    if throttle>=0:
                        throttle = 0
                    else:
                        throttle = throttle-((1+throttle)/2)
                else: #Detected object is far 
                    print("JUST DRIVE")
                    brake=0
                    if throttle>=0:
                        throttle = 1
                    else:
                        throttle = 0
                print('Final throttle = ', throttle)
                print('Final steering = ', steering_angle)

                control_signal = carla.VehicleControl(throttle=throttle, steer=np.float64(steering_angle), brake=brake)
                print(vehicle.get_control())
                vehicle.apply_control(control_signal)
            
        image_queue = queue.LifoQueue()
        camera.listen(image_queue.put)
        vehicle_lead.set_autopilot(True)

        while True:
            world.tick()
            if not image_queue.empty():
                cameraListener(image_queue.get())

    finally:
        print('destroying actors')
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        #collects saved images into a folder
        for result in images:
            result.save()

if __name__ == '__main__':

    main()
