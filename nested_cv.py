import os
import pathlib
import json
import random
import gc
from types import SimpleNamespace
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import KFold


# Functions
def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == ['Cats','Dogs']
    # Integer encode the label
    return tf.argmax(one_hot)


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [256, 256])


def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# Configure dataset for performance
def configure_for_performance(ds):
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def prepare_data(data, train_index, test_index):

    # Get the paths to the data
    train_paths = np.asarray(data)[train_index]
    test_paths = np.asarray(data)[test_index]

    # Make it tf.data.Dataset
    train = tf.data.Dataset.from_tensor_slices(train_paths)
    test = tf.data.Dataset.from_tensor_slices(test_paths)

    # Get labels
    train = train.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    test = test.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    # Configure for performance
    train = configure_for_performance(train)
    test = configure_for_performance(test)

    return train, test


def subsampling(dataset):
    random.shuffle(dataset)
    subsampled_dataset = random.sample(dataset, round(len(dataset)/2))
    return subsampled_dataset


def zero_one_loss(dataset, dataset_size):
    
    _, accuracy = model.evaluate(dataset)
    zero_one_loss = dataset_size*(1-accuracy)

    return int(round(zero_one_loss, ndigits=0))
    
def save_results(results):
    with open('../results.json', 'w') as f:
        json.dump(results, f, indent=4)

# Define the directory of the dataset
data_dir = pathlib.Path('../CatsDogs/')

# Remove corrupted files
os.system("rm CatsDogs/Cats/666.jpg CatsDogs/Dogs/11702.jpg CatsDogs/Dogs/11410.jpg")

# Collects the path of all the files within the dataset
data_paths = [str(path) for path in list(data_dir.glob("*/*.jpg"))]

# Convert non-jpeg images into jpeg files
formats = [(path, Image.open(path).format) for path in data_paths]
non_jpegs = list(filter(lambda x: x[1]!='JPEG', formats))
for path, _ in non_jpegs:
    img = Image.open(path)
    img.convert('RGB').save(path, format='JPEG')

# Fixed hyper-paramters
batch_size = 64
# Hypter-parameters to by tuned
filters_coeffs = ['same', 'incremental']
list_n_filters = [16, 32, 64]
kernel_sizes = [3, 5, 7]
list_n_epochs = [10, 15, 20]


# Run nested cross validation to find the best hyper-parameters for this architecture
# Nested cross-val
subsampled_data_paths = subsampling(data_paths)
k_fold = KFold(n_splits=5)
k_splits = list(k_fold.split(subsampled_data_paths))

# Compute training part
train_index, _ = k_splits[0]
train_paths = np.asarray(subsampled_data_paths)[train_index]

best_model = {}
internal_results=[]
internal_count=0
# Internal cross validation
for filters_coeff in filters_coeffs:
    for n_filters in list_n_filters:
        for kernel_size in kernel_sizes:
            for n_epochs in list_n_epochs:
                internal_splits = k_fold.split(train_paths)
                tmp_results = []
                
                for internal_train_index, internal_test_index in internal_splits:  # Interal cross validation
                    internal_train, internal_test = prepare_data(train_paths, internal_train_index, internal_test_index)

                    model = tf.keras.Sequential([
                        tf.keras.layers.Rescaling(1./255),
                        tf.keras.layers.Conv2D(n_filters, kernel_size, activation=tf.nn.relu, input_shape=(256, 256 ,3)),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Conv2D(n_filters * (1, 2)[filters_coeff=='incremental'], kernel_size, activation=tf.nn.relu),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Conv2D(n_filters * (1, 4)[filters_coeff=='incremental'], kernel_size, activation=tf.nn.relu),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(256, activation=tf.nn.relu),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])

                    model.compile(
                        optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )

                    history = model.fit(
                        internal_train,
                        epochs=n_epochs,
                        verbose=0
                    )

                    internal_loss = zero_one_loss(internal_test, len(internal_test_index))
                    tmp_results.append(internal_loss)
                    
                    # Clear the model
                    del model
                    tf.keras.backend.clear_session()
                    gc.collect()

                result = {'filters_coeff': filters_coeff, 
                          'n_filters': n_filters, 
                          'kernel_size': kernel_size,
                          'n_epochs': n_epochs,
                          'zero_one_loss': np.round(np.mean(tmp_results), decimals=0)}  # Compute the mean loss of the internal cv

                internal_results.append(result)
                save_results(internal_results)
                print(f"Finished internal iteration {internal_count}")
                internal_count+=1

best_model = min(internal_results, key=lambda x: x['zero_one_loss'])
print(f'Best hyper-parameters: {best_model}')
