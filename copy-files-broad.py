import os
import shutil

src = 'D:\Downloads\ocr_data'

try:
    os.mkdir('broad-dataset')
except:
    print('Deleting previous contents')
    for directory in os.scandir('broad-dataset'):
        shutil.rmtree(directory.path)

folders = ['digit', 'upper', 'lower']
counts = {key: 0 for key in folders}

print('Creating folders')
for folder in folders:
    os.mkdir('broad-dataset/{}'.format(folder))

for directory in os.scandir(src):
    print('Copying contents of {}'.format(directory.name))
    for inner_directory in os.scandir(directory):
        for inner_inner_directory in os.scandir(inner_directory):
            for item in os.scandir(inner_inner_directory):
                folder = 'upper'
                if inner_inner_directory.name.isdigit():
                    folder = 'digit'
                elif inner_inner_directory.name.islower():
                    folder = 'lower'

                shutil.copyfile(item.path, 'broad-dataset/{}/{}.png'.format(folder, counts[folder]))
                counts[folder] += 1