import os
import moviepy.video.io.ImageSequenceClip as cvrt

image_folder='images'
fps=10

image_files = [os.path.join(image_folder,img)
               for img in os.listdir(image_folder)
               if img.endswith(".jpg")]
clip = cvrt(image_files, fps=fps)
clip.write_videofile('video.mp4')