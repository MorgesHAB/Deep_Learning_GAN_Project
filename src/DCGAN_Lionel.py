'''
 ______      ______    ______       _       ____  _____
|_   _ `.  .' ___  | .' ___  |     / \     |_   \|_   _|
  | | `. \/ .'   \_|/ .'   \_|    / _ \      |   \ | |
  | |  | || |       | |   ____   / ___ \     | |\ \| |
 _| |_.' /\ `.___.'\\ `.___]  |_/ /   \ \_  _| |_\   |_
|______.'  `.____ .' `._____.'|____| |____||_____|\____|

\brief DD2424 Deep Learning Project in KTH
       Implementation and parameters analysis of a
       Deep Convolutional Generative Adversarial Network (DCGAN)

\author      ISOZ Lionel | EPFL-KTH
\date        19.05.2021
'''
###################################################################
# Libraries importation
# Framework Keras from Tensorflow is used
import tensorflow as tf
from tensorflow.keras import layers

# Basic libraries used in Deep Learning
import numpy as np
import matplotlib.pyplot as plt


###################################################################
# Dataset importation
def import_MNIST():
    # MNIST dataset available directly from TensorFlow keras
    # (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # MNIST fashion
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Expand to 3D, reshape
    X_train = np.expand_dims(X_train, axis=-1)  # For MNIST: (60000, 28, 28, 1)
    # Normalize the images to [-1, 1]
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    #X_train = X_train.astype(np.float32) / 255.0  # for Generator 2
    return X_train


###################################################################
# Generator
def myGenerator1(latent_dim=100):
    model = tf.keras.models.Sequential([
        layers.Dense(7 * 7 * 128, input_shape=[latent_dim]),
        layers.Reshape([7, 7, 128]),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, (5, 5), (2, 2), padding="same", activation="selu"),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(1, (5, 5), (2, 2), padding="same", activation="tanh")
    ])
    return model


def myGenerator2(latent_dim=100):
    model = tf.keras.models.Sequential([
        # foundation for 7x7 image
        layers.Dense(128 * 7 * 7, input_dim=latent_dim),
        # layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, 128)),
        # upsample to 14x14
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        # layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        # upsample to 28x28
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        # layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same')
    ])
    return model


def myGenerator3(latent_dim=100):
    model = tf.keras.models.Sequential([
        # foundation for 7x7 image
        layers.Dense(256 * 7 * 7, input_dim=latent_dim, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Reshape((7, 7, 256)),
        # upsample to 7 x 7 x 128
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # upsample to 14 x 14 x 64
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # Final to 28 x 28 x 1
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model


def GeneratorPaper(latent_dim=100):
    model = tf.keras.models.Sequential([
        # foundation for 7x7 image
        layers.Dense(256 * 7 * 7, input_dim=latent_dim, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),

        layers.Reshape((7, 7, 256)),
        # upsample to 7 x 7 x 128
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),

        # upsample to 14 x 14 x 64
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),

        # Final to 28 x 28 x 1
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model


# generator = myGenerator();
# generator.summary()

###################################################################
# GAN Adversarial model, for updating the generator
def GAN_model(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # Combine both generator and discriminator
    GAN = tf.keras.models.Sequential([generator, discriminator])
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    # opt = tf.keras.optimizers.Adam(2e-4)
    # opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=3e-8)
    GAN.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
    return GAN


###################################################################
# Train model
# This annotation causes the function to be "compiled".
# @tf.function
def train(GAN, dataset, n_epochs=100, batch_size=256, latent_dim=100, img_nbr=100):
    generator, discriminator = GAN.layers
    # keep same seed to generate image (useful for GIF)
    # X_latent_img = tf.random.normal(shape =[img_nbr, latent_dim])

    nbr_batches = int(dataset.shape[0] / batch_size)
    for epoch in range(1, n_epochs + 1):
        for j in range(nbr_batches):

            # Train Discriminator model
            X_latent = tf.random.normal(shape=[batch_size // 2, latent_dim])
            X_fake = generator(X_latent)  # prediction
            y_fake = np.zeros((batch_size // 2, 1))
            X_real = dataset[np.random.randint(0, dataset.shape[0], batch_size // 2)]
            y_real = np.ones((batch_size // 2, 1))
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))

            discriminator.trainable = True
            d_loss, d_acc = discriminator.train_on_batch(X, y)

            # Train Adversarial model
            X_latent2 = tf.random.normal(shape=[batch_size, latent_dim])
            y_fool = np.ones((batch_size, 1))  # create inverted labels for the fake samples

            discriminator.trainable = False
            gan_loss, gan_acc = GAN.train_on_batch(X_latent2, y_fool)

            # Print result
            print(f"Epoch {epoch} Batch {j}/{nbr_batches} Discriminator: Loss: {d_loss:0.3f} Acc: {d_acc:2.2%}  GAN: Loss: {gan_loss:0.3f}, Acc {gan_acc:2.2%}")

        if epoch % 20 == 0:
            discriminator.save('discri_epoch_{:04d}'.format(epoch))

        # Save image generation process after each epoch
        X_latent_img = tf.random.normal(shape=[img_nbr, latent_dim])
        save_generated_img(generator, epoch, X_latent_img)

    ###################################################################


# Save images generated by the GAN
def save_generated_img(generator, epoch, X_latent):
    if epoch % 10 == 0:
        generator.save('generator_epoch_{:04d}'.format(epoch))

    X_generated = generator(X_latent, training=False)

    # fig = plt.figure(figsize =(10, 10)) ### Warining never close figure
    for i in range(6 * 6):
        plt.subplot(6, 6, i + 1)
        plt.imshow(X_generated[i, :, :, 0], cmap='gray_r')
        plt.axis('off')

    plt.savefig('image_epoch_{:04d}.png'.format(epoch))
    plt.show()

###################################################################
# Discriminator
def myDiscriminator(input_shape=(28, 28, 1), dropout=0.3):
    model = tf.keras.models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=input_shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(dropout),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        # layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(dropout),

        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    # compile model
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    # opt = tf.keras.optimizers.Adam(2e-4)
    # RMSProp as optimizer generates more realistic fake images compared to Adam for this case
    # opt = tf.keras.optimizers.RMSprop(lr=0.0002, decay=6e-8)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


###################################################################
# Generator
def myGenerator0(latent_dim=100):
    model = tf.keras.models.Sequential([
        layers.Dense(7 * 7 * 128, input_shape=[latent_dim]),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0),  ## relu

        layers.Reshape([7, 7, 128]),
        layers.Conv2DTranspose(64, (5, 5), (2, 2), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0),  #### relu

        layers.Conv2DTranspose(1, (5, 5), (2, 2), padding="same", activation="tanh")
        # Batch Norm instability was avoided by not applying batchnorm to the generator output layer
    ])
    return model


# Generator
def myGenerator3L(latent_dim=100):
    model = tf.keras.models.Sequential([
        layers.Dense(7 * 7 * 256, input_shape=[latent_dim]),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0),  ## relu

        layers.Reshape([7, 7, 256]),
        layers.Conv2DTranspose(128, (5, 5), (1, 1), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0),  #### relu

        layers.Conv2DTranspose(64, (5, 5), (2, 2), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0),  #### relu

        layers.Conv2DTranspose(1, (5, 5), (2, 2), padding="same", activation="tanh", use_bias=False)
        # Batch Norm instability was avoided by not applying batchnorm to the generator output layer
    ])
    return model


###################################################################
# Main Program

dataset = import_MNIST()

generator = myGenerator0()
discriminator = myDiscriminator()
GAN = GAN_model(generator, discriminator)

train(GAN, dataset, n_epochs=100, batch_size=64)
