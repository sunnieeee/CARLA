'''
Convert images in the 'images' folder to a video.
Created by Yuqing Cao.
Dated May/17/2020
'''

import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip as img2vid

# images path
dirname = os.path.dirname(__file__)
image_folder = os.path.join(dirname,'../images')
fps = 10

image_files = [
    os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".jpg")
]

clip = img2vid(image_files, fps)
# video path
clip.write_videofile(os.path.join(dirname,'../video.mp4')) # change video name here
