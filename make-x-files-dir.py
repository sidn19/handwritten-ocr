import os
import shutil
import random

X = 2000

try:
    os.mkdir('isochronous-dataset')
except:
    print('Deleting previous contents')
    for directory in os.scandir('isochronous-dataset'):
        shutil.rmtree(directory.path)

for directory in os.scandir('dataset'):
    print('Copying {} random images for class {}'.format(X, directory.name))
    os.mkdir('isochronous-dataset/{}'.format(directory.name))
    files = []
    for item in os.scandir(directory):
        files.append(os.path.join(directory.path, item.name))
    
    files = random.sample(files, min(500, len(files)))
    for index, file_path in enumerate(files):
        shutil.copyfile(file_path, 'isochronous-dataset/{}/{}.png'.format(directory.name, index))