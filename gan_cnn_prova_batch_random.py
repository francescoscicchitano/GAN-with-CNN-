from keras.datasets import mnist

from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten
from keras.layers import Input, LeakyReLU, Reshape, UpSampling2D, ZeroPadding2D
from keras.models import Model, Sequential

import matplotlib.pyplot as plt

from models.gan import GAN

import numpy as np
import pickle
from datetime import datetime


class GAN_CNN(GAN):
    def __init__(self, parameters):
        super().__init__(parameters)

    def _inner_discriminator(self):
        model = Sequential()

        model.add(Conv2D(self.parameters.d_filters[0], kernel_size=self.parameters.d_kernel_size,
                         strides=self.parameters.d_strides, input_shape=self.parameters.real_data_shape,
                         padding=self.parameters.d_padding))
        model.add(LeakyReLU(self.parameters.d_alpha))
        model.add(Dropout(self.parameters.d_dropout))
        model.add(Conv2D(self.parameters.d_filters[1], kernel_size=self.parameters.d_kernel_size,
                         strides=self.parameters.d_strides, padding=self.parameters.d_padding))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=self.parameters.g_momentum))
        model.add(LeakyReLU(self.parameters.d_alpha))
        model.add(Dropout(self.parameters.d_dropout))
        model.add(Conv2D(self.parameters.d_filters[2], kernel_size=self.parameters.d_kernel_size,
                         strides=self.parameters.d_strides, padding=self.parameters.d_padding))
        model.add(BatchNormalization(momentum=self.parameters.g_momentum))
        model.add(LeakyReLU(self.parameters.d_alpha))
        model.add(Dropout(self.parameters.d_dropout))
        model.add(Conv2D(self.parameters.d_filters[3], kernel_size=self.parameters.d_kernel_size,
                         strides=self.parameters.d_last_strides, padding=self.parameters.d_padding))
        model.add(BatchNormalization(momentum=self.parameters.g_momentum))
        model.add(LeakyReLU(self.parameters.d_alpha))
        model.add(Dropout(self.parameters.d_dropout))
        model.add(Flatten())
        model.add(Dense(1, activation=self.parameters.d_last_activation))

        img = Input(shape=self.parameters.real_data_shape)
        validity = model(img)

        return Model(img, validity, name="GAN_CNN_Discriminator")


    def _inner_generator(self):

        model = Sequential()

        model.add(Dense(self.parameters.d_filters[2] * self.parameters.g_dense_size  * self.parameters.g_dense_size, activation=self.parameters.g_activation, input_dim=self.parameters.noise_data_shape))
        model.add(Reshape((self.parameters.g_dense_size, self.parameters.g_dense_size, self.parameters.d_filters[2])))
        model.add(UpSampling2D())
        model.add(Conv2D(self.parameters.d_filters[2], kernel_size=self.parameters.d_kernel_size, padding=self.parameters.d_padding))
        model.add(BatchNormalization(momentum=self.parameters.g_momentum))
        model.add(Activation(self.parameters.g_activation))
        model.add(UpSampling2D())
        model.add(Conv2D(self.parameters.d_filters[1], kernel_size=self.parameters.d_kernel_size, padding=self.parameters.d_padding))
        model.add(BatchNormalization(momentum=self.parameters.g_momentum))
        model.add(Activation(self.parameters.g_activation))
        model.add(Conv2D(self.parameters.g_channels, kernel_size=self.parameters.d_kernel_size, padding=self.parameters.d_padding))
        model.add(Activation(self.parameters.g_last_activation))

        noise = Input(shape=(self.parameters.noise_data_shape,))
        img = model(noise)

        return Model(noise, img, name="GAN_CNN_Generator")



    def _inner_adversarial(self):
        z = Input(shape=(self.parameters.noise_data_shape,))
        img = self.generator(z)  # self.generator

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.adversarial = Model(z, valid)
        self.adversarial.compile(loss='binary_crossentropy', optimizer=self.parameters.optimizer, metrics=['accuracy'])

        return self.adversarial


    def _generator_noise(self, data_size):
        return np.random.normal(0.0, 1.0, size=(data_size, self.parameters.noise_data_shape))


    def train(self, data):
        start_train = datetime.now()
        outliers = list()

        label_valid = np.ones((self.parameters.batch_size, 1))
        label_fake = np.zeros((self.parameters.batch_size, 1))

        for epoch in range(self.parameters.n_epochs):
            start_epoch = datetime.now()

            # ### Train the discriminator ### #

            batch_idx = np.random.randint(0, data.shape[0], self.parameters.batch_size)

            imgs = data[batch_idx]

            noise = self._generator_noise(self.parameters.batch_size)
            gen_imgs = self.generator.predict(noise)

            self.discriminator.trainable = True

            loss1, acc1 = self.discriminator.train_on_batch(imgs, label_valid)
            loss2, acc2 = self.discriminator.train_on_batch(gen_imgs, label_fake)

            loss = 0.5 * np.add(loss1, loss2)
            acc = 0.5 * np.add(acc1, acc2)

            self.discriminator.trainable = False

            g_loss, g_acc = self.adversarial.train_on_batch(noise, label_valid)



            log = "\tEpoch: %d [discriminator loss: %f, acc: %f%%] [G loss: %f, acc.: %.2f%%] time: %s" % (
                epoch, loss, 100 * acc, g_loss, 100 * g_acc,
                (datetime.now().timestamp()) - (start_epoch.timestamp())
            )
            print(log)


            if epoch >= self.parameters.burnin and epoch % self.parameters.save_steps == 0:
                outliers = self.outliers_plot(epoch)

            print("Execution time for epoch: %s" % (datetime.now().timestamp() - start_epoch.timestamp()))


        with open(self.parameters.gan_outliers_prova_batch_random, 'wb') as file_handler:
            pickle.dump(outliers, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

        print("Training time: %s" % (datetime.now().timestamp() - start_train.timestamp()))


    def outliers_plot(self, epoch):
        r, c = 5, 5
        outliers = list()
        noise = np.random.normal(0, 1, (r * c, self.parameters.noise_data_shape))

        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        outliers.append(gen_imgs)

        outliers = np.array(outliers)

        fig, axs = plt.subplots(r, c)
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig(self.parameters.plot_folder_prova_cnn_batch_random + "mnist_%d.png" % epoch)
        plt.close()
        return outliers

if __name__ == '__main__':
    from model_parameters import gan_cnn_parameters as p

    np.random.seed(101)

    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    gan_cnn = GAN_CNN(p)
    gan_cnn.build()

    gan_cnn.train(X_train)