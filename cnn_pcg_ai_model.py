import os
import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential  # type: ignore
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense  # type: ignore
from keras.utils import to_categorical  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import matplotlib.pyplot as plt

# Set the path to the dataset folder
dataset_path = '/Users/juhidwivedi/Desktop/CNN_PCG/dataset'

# Define the classes
classes = ['normal', 'aortic_stenosis', 'mitral_stenosis', 'mitral_valve_prolapse', 'pericardial_murmurs']

# Data preprocessing
pcg_data = []
labels = []
for i, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path, class_name)
    for filename in os.listdir(class_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(class_path, filename)
            _, signal = wavfile.read(file_path)
            pcg_data.append(signal)
            labels.append(i)

# Normalize each individual signal in pcg_data
scaler = MinMaxScaler()
normalized_data = []

for signal in pcg_data:
    normalized_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    normalized_data.append(normalized_signal)

# Convert the normalized_data list to a NumPy array
normalized_data = np.array(normalized_data, dtype=object)
labels = np.array(labels)

# Outlier Removal
filtered_data = []
filtered_labels = []
threshold = 0.1  # Adjust the threshold value as needed

for i, pcg_signal in enumerate(normalized_data):
    energy = np.sum(pcg_signal ** 2)
    if energy > threshold:
        filtered_data.append(pcg_signal)
        filtered_labels.append(labels[i])

# Scaling the filtered data
scaler = MinMaxScaler()
scaled_data = []

for pcg_signal in filtered_data:
    scaled_signal = scaler.fit_transform(pcg_signal.reshape(-1, 1)).flatten()
    scaled_data.append(scaled_signal)

# Convert the scaled_data list to a NumPy array
scaled_data = np.array(scaled_data, dtype=object)
filtered_labels = np.array(filtered_labels)

# Data augmentation
data_augmented = []
labels_augmented = []
augmentation_factor = 3  # Number of augmented samples to generate for each original sample
max_length = 1000  # Adjust the maximum length as needed

for i in range(len(scaled_data)):
    x = scaled_data[i]
    y = filtered_labels[i]

    # Perform data augmentation
    for _ in range(augmentation_factor):
        # Time Shift
        time_shift = np.random.randint(-50, 50)  # Specify the time shift range
        x_shifted = np.roll(x, time_shift)

        # Pad the sequence
        x_padded = pad_sequences([x_shifted], maxlen=max_length, dtype='float32', padding='post', truncating='post')[0]

        # Add the augmented sample and its label to the augmented data list
        data_augmented.append(x_padded)
        labels_augmented.append(y)

# Convert the augmented data to numpy arrays
data_augmented = np.array(data_augmented)
labels_augmented = np.array(labels_augmented)

# Shuffle the augmented data
data_augmented, labels_augmented = shuffle(data_augmented, labels_augmented, random_state=42)

# Reshape the data for CNN input
data_augmented = data_augmented.reshape(data_augmented.shape[0], max_length, 1)

# Define training parameters
epochs = 50
batch_size = 32

# Data splitting
x_train, x_test, y_train, y_test = train_test_split(data_augmented, labels_augmented, test_size=0.2, random_state=42)

# Print shapes to verify
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Prepare the CNN model architecture
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=10, activation='relu', input_shape=(max_length, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=len(classes), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, to_categorical(y_train), epochs=epochs, batch_size=batch_size, validation_data=(x_test, to_categorical(y_test)))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, to_categorical(y_test))
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Save the model
model.save('pcg_model.h5')
