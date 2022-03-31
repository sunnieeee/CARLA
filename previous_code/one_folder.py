import os
from PIL import Image

ctr = 0
paths = []
for root, subdirs, files in os.walk('runs/detect/'):
    for file in files:
        paths.append(os.path.join(root, file))
        
for path in sorted(paths, key=lambda x: len(x)):
    print(path)
    img = Image.open(path)
    img.save(os.path.join("images", "img_"+str(ctr)+".jpg"))
    ctr+=1
os.remove('runs/')
