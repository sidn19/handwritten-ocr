# %%
from keras.preprocessing import image_dataset_from_directory
from tensorflow import data
AUTOTUNE = data.AUTOTUNE

image_height = 64
image_width = 64
batch_size = 32

dataset = 'isochronous-dataset'

X_train = image_dataset_from_directory(dataset, validation_split=0.2, subset='training', seed=0, image_size=(image_height, image_width), color_mode='grayscale', batch_size=batch_size)
X_test = image_dataset_from_directory(dataset, validation_split=0.2, subset='validation', seed=0, image_size=(image_height, image_width), color_mode='grayscale', batch_size=batch_size)

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
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import regularizers

num_classes = len(class_names)

model = Sequential([
  Rescaling(1.0 / 255, input_shape=(image_height, image_width, 1)),
  
  Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
  BatchNormalization(),
  MaxPooling2D((2, 2)),

  Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
  BatchNormalization(),
  Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
  BatchNormalization(),
  MaxPooling2D((2, 2)),

  Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
  BatchNormalization(),
  Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
  BatchNormalization(),
  MaxPooling2D((2, 2)),

  Flatten(),
  Dense(200, activation='relu', kernel_initializer='he_normal'),
  Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# %%
history = model.fit(X_train, validation_data=X_test, epochs=10, use_multiprocessing=True, workers=8, max_queue_size=500)

def print_to_file(s):
    with open('modelsummary.txt','a') as f:
        print(s, file=f)

model.summary(print_fn=print_to_file)

with open('modelsummary.txt','a') as f:
  print('Accuracy: {}'.format(history.history['accuracy']), file=f)

# %%
model.save(dataset + '.model')

# %%
from tensorflow.keras.models import load_model

loaded_model = load_model(dataset + '.model')
X_pred = image_dataset_from_directory('predict', seed=0, image_size=(image_height, image_width), shuffle=False, color_mode='grayscale')
y_pred = loaded_model.predict_classes(X_pred)
y_pred = list(map(lambda x: class_names[x], y_pred.tolist()))
print(y_pred)

# %%
