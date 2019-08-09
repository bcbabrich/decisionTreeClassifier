print('image classifier')

from matplotlib.image import imread
from datetime import datetime
startTime = datetime.now()


import os

directory = 'C:\\Users\\bbabrich\\Downloads\\severstal-steel-defect-detection\\train_images'
data = []
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        img = imread(os.path.join(directory, filename))
        img = img.flatten()
        data.append(img)
        continue
    else:
        continue
print(datetime.now() - startTime)
