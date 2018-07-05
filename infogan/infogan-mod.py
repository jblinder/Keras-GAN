from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.utils.training_utils import multi_gpu_model # https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/
import keras.backend as K

import pickle
import keras

import matplotlib.pyplot as plt
import numpy as np
import os
import logging

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

class INFOGAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.num_classes = 40
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 102

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', self.mutual_info_loss]

        # Build and the discriminator and recognition network
        self.discriminator, self.auxilliary = self.build_disk_and_q_net()

        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the recognition network Q
        self.auxilliary.compile(loss=[self.mutual_info_loss],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        gen_input = Input(shape=(self.latent_dim,))
        img = self.generator(gen_input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        valid = self.discriminator(img)
        # The recognition network produces the label
        target_label = self.auxilliary(img)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(gen_input, [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)


        logging.basicConfig(filename='build.log',level=logging.DEBUG)

    def build_generator(self):

        model = Sequential()
        
        # If using 12x12 then use 3 conv blocks, if using 6x6 then use 4 conv blocks 
        # (Can change xs to dims (i.e. 4) -- divide imgs by num like 16. It's easier with square images)
        
        
        # Modified CNN
        model.add(Dense(128 * 4 * 4, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((4, 4, 128)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(3, kernel_size=4, padding='same'))
        model.add(Activation("tanh"))
        
        gen_input = Input(shape=(self.latent_dim,))

        img = model(gen_input)
        print("gen summary: ")
        model.summary()

        return Model(gen_input, img)


    def build_disk_and_q_net(self):
        img = Input(shape=self.img_shape)

        # Shared layers between discriminator and recognition network
        model = Sequential()
        # Previous CNN
        
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same")) #Changed this to speed up build time #model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
       
        model.add(Flatten())
        
        img_embedding = model(img)

        # Discriminator
        validity = Dense(1, activation='sigmoid')(img_embedding)

        # Recognition
        q_net = Dense(128, activation='relu')(img_embedding)

        # NOTE: Logit/ Signmoid for activation? Soph said to change this
        label = Dense(self.num_classes, activation='softmax')(q_net)

        disk = Model(img, validity)
        
        print("disk summary")
        disk.summary()
        
        q = Model(img, label)
    
        print("q summary")
        q.summary()

        # Return discriminator and recognition network
        return Model(img, validity), Model(img, label)


    def mutual_info_loss(self, c, c_given_x):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return conditional_entropy + entropy

    def sample_generator_input(self, batch_size):
        # Generator inputs
        sampled_noise = np.random.normal(0, 1, (batch_size, 62))

        sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
        sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)
  
        return sampled_noise, sampled_labels
    
    # sample_interval to higher
    # save model for every sample interval -- so you can always go back to checkpoints
    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, y_train),(_, _) = pickle.load( open( "celebA-dset-resize-square-128-200000.pkl", "rb" ) ) #mnist.load_data()
        X_train = np.array(X_train)
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5 # Rescale -1 to 1
        y_train = y_train.reshape(-1, 1)
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and categorical labels
            # JUSTIN: Confused
            sampled_noise, sampled_labels = self.sample_generator_input(batch_size)
            gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)
    
            # Generate a half batch of new images
            self.generator.summary()
            
            gen_imgs = self.generator.predict(gen_input)
            # Train on real and generated data

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            # Avg. loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator and Q-network
            # ---------------------

            g_loss = self.combined.train_on_batch(gen_input, [valid, sampled_labels])

            # Plot the progress
            print ("%d:[DLoss:%.2f,Acc:%.2f%%]:[QLoss:%.2f]:[GLoss: %.2f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))
            logging.info("%d:[DLoss:%.2f,Acc:%.2f%%]:[QLoss:%.2f]:[GLoss: %.2f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
    
    '''
    Used to generate predictions when model is complete
    '''
    def sample_images(self, epoch):
        '''
        Plot grid of faces
        '''
        r, c = 5, 5

        fig, axs = plt.subplots(r, c,figsize=(12,12),dpi=200)
        for i in range(c):
            sampled_noise, _ = self.sample_generator_input(c)
            # turns into one-hot-vector 
            #(don't need this) -- pass in 40 dim for each row
            label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=self.num_classes)

            logging.info(f'Epoch: {epoch} Label: {label}')
            
            gen_input = np.concatenate((sampled_noise, label), axis=1)
            gen_imgs = self.generator.predict(gen_input)
            gen_imgs = 0.5 * gen_imgs + 0.5

            for j in range(r):
                axs[j,i].imshow(gen_imgs[j,:,:])
                axs[j,i].axis('off')

        fig.savefig("images/%d.png" % epoch, bbox_inches='tight')
        plt.close()
        
        '''
        Plot individual face
        '''
        fig, axs = plt.subplots(1,1,figsize=(12,12),dpi=200)
        sampled_noise, _ = self.sample_generator_input(c)
        # turns into one-hot-vector (don't need this) -- pass in 40 dim for each row
        label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=self.num_classes)

        logging.info(f'Epoch: {epoch} Label: {label}')

        gen_input = np.concatenate((sampled_noise, label), axis=1)
        gen_imgs = self.generator.predict(gen_input)
        gen_imgs = 0.5 * gen_imgs + 0.5
        axs.imshow(gen_imgs[j,:,:])
        axs.axis('off')
            
        fig.savefig("images/face-%d.png" % epoch, bbox_inches='tight')
        plt.close()


    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])
            model.save(model_name+'.h5')
        save(self.generator, "generator")
        save(self.discriminator, "discriminator")

if __name__ == '__main__':
    infogan = INFOGAN()
    infogan.train(epochs=50000, batch_size=164, sample_interval=200)
    infogan.save_model()

