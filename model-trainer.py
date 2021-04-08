# %%
from keras.preprocessing import image_dataset_from_directory
from tensorflow import data
AUTOTUNE = data.AUTOTUNE

image_height = 64
image_width = 64

X_train = image_dataset_from_directory('isochronous-dataset', validation_split=0.2, subset='training', seed=0, image_size=(image_height, image_width), label_mode='categorical', color_mode='grayscale')
X_test = image_dataset_from_directory('isochronous-dataset', validation_split=0.2, subset='validation', seed=0, image_size=(image_height, image_width), label_mode='categorical', color_mode='grayscale')

for image_batch, labels_batch in X_train:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

class_names = X_train.class_names
print(class_names)

# Prefetch to memory
X_train = X_train.cache().prefetch(buffer_size=AUTOTUNE)
X_test = X_test.cache().prefetch(buffer_size=AUTOTUNE)

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.experimental.preprocessing import Rescaling
from keras.optimizers import SGD
from tensorflow.keras import regularizers

num_classes = 62

model = Sequential([
  Rescaling(1.0 / 255, input_shape=(image_height, image_width, 1)),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  Conv2D(64, (3, 3), activation='relu'),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Flatten(),
  Dense(100, activation='relu',
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
  ),
  Dense(num_classes, activation='softmax')
])

optimizer = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# %%
model.fit(X_train, validation_data=X_test, epochs=10)
# %%
X_pred = image_dataset_from_directory('predict', seed=0, image_size=(image_height, image_width), label_mode='categorical', shuffle=False, color_mode='grayscale')
y_pred = model.predict_classes(X_pred)
y_pred = list(map(lambda x: class_names[x], y_pred.tolist()))
print(y_pred)
# %%
