import os
from PIL import Image
import shutil

# sort images in runs folder
paths = []
dirname = os.path.dirname(__file__)
run_pth = os.path.join(dirname,'../runs/detect/')
if not os.path.exists(run_pth):
    raise Exception("Runs folder does not exist!")
for root, subdirs, files in os.walk(run_pth):
    for file in files:
        paths.append(os.path.join(root, file))

# save sorted images in images folder
ctr = 0        
for path in sorted(paths, key=lambda x: len(x)):
    print(path)
    img = Image.open(path)
    pth = os.path.join(dirname,'../images')
    if not os.path.exists(pth):
        os.makedirs(pth)
    img.save(os.path.join(pth, "img_"+str(ctr)+".jpg"))
    ctr+=1

# delete runs folder
shutil.rmtree(os.path.join(dirname,'../runs'))
