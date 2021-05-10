# %%
from keras.preprocessing import image_dataset_from_directory
from tensorflow import data
AUTOTUNE = data.AUTOTUNE

image_height = 64
image_width = 64
batch_size = 32

dataset = 'isochronous-dataset'

X_raw_train = image_dataset_from_directory(dataset, validation_split=0.2, subset='training', seed=0, image_size=(image_height, image_width), color_mode='grayscale', batch_size=batch_size)
X_raw_test = image_dataset_from_directory(dataset, validation_split=0.2, subset='validation', seed=0, image_size=(image_height, image_width), color_mode='grayscale', batch_size=batch_size)


class_names = X_raw_train.class_names
print(class_names)
# %%
import numpy as np

X_train = []
X_test = []
y_train = []
y_test = []
for image_batch, labels_batch in X_raw_train:
    for i, image in enumerate(image_batch):
        arr = []
        for a in image_batch[i]:
            arr.append(a.numpy().tolist()[0][0])
        X_train.append(arr)
        y_train.append(labels_batch[i].numpy())

for image_batch, labels_batch in X_raw_test:
    for i, image in enumerate(image_batch):
        arr = []
        for a in image_batch[i]:
            arr.append(a.numpy().tolist()[0][0])
        X_test.append(arr)
        y_test.append(labels_batch[i].numpy())

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

# %%
print('Number of mislabeled points out of a total %d points : %d' % (X_test.shape[0], (y_test != y_pred).sum()))



# %%
