import os
import shutil

src = 'D:\Downloads\ocr_data'

only_charas = True

try:
    os.mkdir('dataset')
except:
    print('Deleting previous contents')
    for directory in os.scandir('dataset'):
        shutil.rmtree(directory.path)

folders = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]

if not only_charas:
    folders += [str(i) for i in range(10)]

counts = [0] * len(folders)

print('Creating folders')
for folder in folders:
    os.mkdir('dataset/{}'.format(folder))

for directory in os.scandir(src):
    print('Copying contents of {}'.format(directory.name))
    for inner_directory in os.scandir(directory):
        for index, inner_inner_directory in enumerate(os.scandir(inner_directory)):
            if not only_charas or inner_inner_directory.name.isalpha():
                for item in os.scandir(inner_inner_directory):
                    shutil.copyfile(item.path, 'dataset/{}/{}.png'.format(inner_inner_directory.name, counts[index]))
                    counts[index] += 1