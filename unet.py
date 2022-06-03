import glob
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpi4py import MPI

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

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
batch_size = 100
epochs = 24
shuffle_buffer = 1000
learning_rate = 0.001


## Mapping function for TFrecords
def map_fn_rain_ammount(serialized_example):

    feature_description = {
        'feature': tf.io.FixedLenFeature([len_feature, ], tf.float32),
        'label':     tf.io.FixedLenFeature([len_label, ], tf.float32)
    }

    example = tf.io.parse_single_example(
        serialized_example, feature_description)

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
    inputs = Input((36, 36, 143))

    conv1f = Conv2D(16, 3, padding='same',kernel_initializer='he_normal', activation='relu')(inputs)
    conv1s = Conv2D(16, 3, padding='same',kernel_initializer='he_normal', activation='relu')(conv1f)
    pool1  = MaxPooling2D(pool_size=(2, 2))(conv1s) ## 18x18

    conv2f = Conv2D(32, 3, padding='same',kernel_initializer='he_normal', activation='relu')(pool1)
    conv2s = Conv2D(32, 3, padding='same',kernel_initializer='he_normal', activation='relu')(conv2f)
    pool2  = MaxPooling2D(pool_size=(2, 2))(conv2s) ## 9x9

    conv3f = Conv2D(64, 3, padding='same',kernel_initializer='he_normal', activation='relu')(pool2)
    conv3s = Conv2D(64, 3, padding='same',kernel_initializer='he_normal', activation='relu')(conv3f)

    up9   = concatenate([UpSampling2D(size=(2, 2))(conv3s), pool1], axis=3) ## 18x18
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', activation='relu')(up9)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv9)

    up10  = concatenate([UpSampling2D(size=(2, 2))(conv9), inputs], axis=3) ## 36x36
    conv10 = Conv2D(16, 3, padding='same', kernel_initializer='he_normal', activation='relu')(up10)
    conv10 = Conv2D(16, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv10)

    up11   = UpSampling2D(size=(2,2))(conv10)
    conv11 = Conv2D(8, 3, padding='same', kernel_initializer='he_normal', activation='relu')(up11)
    conv11 = Conv2D(8, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv11)

    outputs = Conv2D(1, 1, activation='relu')(conv11)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer, loss=loss)

        
    ## Train the model
    train_history = model.fit(trn_dataset, epochs=epochs, verbose=0, validation_data=vld_dataset)

    ## Plot loss functions
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    axs.set_ylabel('Loss')
    axs.plot(train_history.history['loss'], label='loss trn', c='tab:blue', alpha=1)
    axs.plot(train_history.history['val_loss'], label='loss val', c='tab:blue', alpha=.3)
    axs.legend()
    fig.savefig('models/loss_' + str(my_rank).zfill(2) + '.png')

    ## Save model and results
    model.save('models/model_' + str(my_rank).zfill(2) + '.h5')
    json.dump(train_history.history, open('models/train_hist_' + str(my_rank).zfill(2) + '.json', 'w'))
    comm.send(1, dest=p-1, tag=123)
