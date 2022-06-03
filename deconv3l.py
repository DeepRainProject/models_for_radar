import glob
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpi4py import MPI

## MPI Parameters
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

res = [72, 72]
output_shape = [72, 72]
len_label = np.prod(res)

input_shape = [36, 36, 143]
len_feature = np.prod(input_shape)

## Hyperparameters
batch_size = 20
epochs = 24
shuffle_buffer = 1000
learning_rate = 0.001


## Mapping function for TFrecords
def map_fn_rain_ammount(serialized_example):

    feature_description = {
        'feature': tf.io.FixedLenFeature([len_feature, ], tf.float32),
        'label':     tf.io.FixedLenFeature([len_label, ], tf.float32)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    feature = example['feature']
    label = example['label']

    feature = tf.reshape(example['feature'], input_shape)
    label = tf.reshape(example['label'], res)

    return (feature, label)


## Master process
if my_rank == p-1:
    
    for i in range(p-1):
        cond = comm.recv(source=i, tag=123)
        print(i, 'sends', cond)
    

else:
    ## Create datasets
    data_path = '/p/scratch/deepacf/deeprain/rojascampos1/data/radar_enhancement/hres/conv_approach/tfrecords'

    trn_path = data_path + '/train_set'
    trn_files = sorted(glob.glob(trn_path + '/*'))
    trn_raw_dataset = tf.data.TFRecordDataset(trn_files)
    trn_raw_dataset = trn_raw_dataset.shuffle(buffer_size=shuffle_buffer)
    trn_dataset = trn_raw_dataset.map(map_fn_rain_ammount)
    trn_dataset = trn_dataset.batch(batch_size=batch_size, drop_remainder=True)
    
    vld_path = data_path + '/validation_set'
    vld_files = sorted(glob.glob(vld_path + '/*'))
    vld_raw_dataset = tf.data.TFRecordDataset(vld_files)
    vld_raw_dataset = vld_raw_dataset.shuffle(buffer_size=shuffle_buffer)
    vld_dataset = vld_raw_dataset.map(map_fn_rain_ammount)
    vld_dataset = vld_dataset.batch(batch_size=batch_size, drop_remainder=True)
    
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredLogarithmicError()

    ## Model
    model = tf.keras.Sequential(name='model')
    model.add(tf.keras.layers.InputLayer(input_shape=(36,36,143)))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size=(5,5), strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=(5,5), strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(1, kernel_size=(5,5), strides=1, padding='same', activation='relu'))
                        
    model.compile(optimizer, loss=loss)
    
    ## Train the model
    train_history = model.fit(trn_dataset, epochs=epochs, verbose=0, validation_data=vld_dataset)

    ## Plot loss functions
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    axs.set_ylabel('Loss')
    axs.plot(train_history.history['loss'], label='loss trn', c='tab:blue', alpha=1)
    axs.plot(train_history.history['val_loss'], label='loss val', c='tab:blue', alpha=.3)
    axs.legend()
    fig.savefig('models/model_loss_' + str(my_rank).zfill(2) + '.png')
    
    ## Save model and results
    model.save('models/model_' + str(my_rank).zfill(2) + '.h5')
    json.dump(train_history.history, open('models/train_hist_' + str(my_rank).zfill(2) + '.json', 'w'))
    comm.send(1, dest=p-1, tag=123)