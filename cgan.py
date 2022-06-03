import glob
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpi4py import MPI

## MPI Parameters
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

res = [72, 72, 1]
output_shape = [72, 72, 1]
len_label = np.prod(res)

input_shape = [36, 36, 143]
len_feature = np.prod(input_shape)

## Hyperparameters
batch_size = 20
epochs = 24
shuffle_buffer = 1000
learning_rate = 0.001
LAMBDA = 100


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
    trn_dataset = trn_raw_dataset.map(map_fn_rain_ammount).shuffle(shuffle_buffer).batch(batch_size, drop_remainder=True)

    # vld_path = data_path + '/validation_set'
    # vld_files = sorted(glob.glob(vld_path + '/*'))
    # vld_raw_dataset = tf.data.TFRecordDataset(vld_files)
    # vld_raw_dataset = vld_raw_dataset.shuffle(buffer_size=shuffle_buffer)
    # vld_dataset = vld_raw_dataset.map(map_fn_rain_ammount)
    # vld_dataset = vld_dataset.batch(batch_size=batch_size, drop_remainder=True)
    
    def make_generator():

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(36,36,143)))

        model.add(tf.keras.layers.MaxPool2D(2))
        
        model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size=(5,5), strides=2, padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=(5,5), strides=2, padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2D(1, kernel_size=(5,5), strides=1, padding='same', activation='relu'))

        return model
    
    def make_discriminator():
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(72,72,1)))

        model.add(tf.keras.layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))

        return model
    
    ## Models
    generator = make_generator()
    discriminator = make_discriminator()


    ## Define loss functions
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    MSLE = tf.keras.losses.MeanSquaredLogarithmicError()

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
 
    def generator_loss(disc_generated_output, gen_output, target):
        gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = MSLE(target, gen_output)
        total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss


    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


    ## Define custom training step
    @tf.function
    def train_step(images):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            generated_images = generator(images[0], training=True)


            real_output = discriminator(images[1], training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output, generated_images, images[1])            
            disc_loss = discriminator_loss(real_output, fake_output)

        ## Apply backpropagation at end of train_step
        gradients_of_generator     = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
        return gen_loss, disc_loss
        
        
    ## Defining train step
    def train(dataset, epochs):
        
        gen_epoch_loss = []
        dis_epoch_loss = []

        for epoch in range(epochs):
            print('epoch', epoch)
            start = time.time()
            
            tstep_gen_loss  = []
            tstep_disc_loss = []
            
            for image_batch in dataset:
                
                loss_g, loss_d = train_step(image_batch)
                tstep_gen_loss.append(loss_g)
                tstep_disc_loss.append(loss_d)
            
            gen_epoch_loss.append(tf.reduce_mean(tstep_gen_loss))
            dis_epoch_loss.append(tf.reduce_mean(tstep_disc_loss))
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            
        return gen_epoch_loss, dis_epoch_loss

    ## Train the models
    generator_loss, discriminator_loss = train(trn_dataset, epochs)

    ## Save models
    generator.save('models/model_' + str(my_rank).zfill(2) + '.h5')
    comm.send(1, dest=p-1, tag=123)
