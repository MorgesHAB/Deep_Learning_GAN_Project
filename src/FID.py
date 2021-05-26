'''
__/\\\\\\\\\\\\\\\__/\\\\\\\\\\\__/\\\\\\\\\\\\____
 _\/\\\///////////__\/////\\\///__\/\\\////////\\\__
  _\/\\\_________________\/\\\_____\/\\\______\//\\\_
   _\/\\\\\\\\\\\_________\/\\\_____\/\\\_______\/\\\_
    _\/\\\///////__________\/\\\_____\/\\\_______\/\\\_
     _\/\\\_________________\/\\\_____\/\\\_______\/\\\_
      _\/\\\_________________\/\\\_____\/\\\_______/\\\__
       _\/\\\______________/\\\\\\\\\\\_\/\\\\\\\\\\\\/___
        _\///______________\///////////__\////////////_____

\brief DD2424 Deep Learning Project in KTH
       Implementation of the Frechet Inception Distance (FID)
       for Evaluating Generative Adversarial Networks (GANs)
       Based on:
       https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

\author      ISOZ Lionel | EPFL-KTH
\date        21.05.2021
'''
###################################################################
# Libraries importation
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import sqrtm
from skimage.transform import resize
from keras.applications.inception_v3 import InceptionV3, preprocess_input

###################################################################
# Scale images in order to use InceptionV3
def scale_images(images, new_shape):
  images_list = list()
  for image in images:
    # resize with nearest neighbor interpolation
    new_image = resize(image, new_shape, 0)
    images_list.append(new_image)
  return np.asarray(images_list)

###################################################################
# Calculate Frechet Inception Distance (FID)
def compute_FID(act1, act2):
  # calculate mean and covariance statistics
  mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
  mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
  # calculate sqrt of product between cov
  covmean = sqrtm(np.dot(sigma1, sigma2))
  # calculate score
  FID = np.dot(mu1 - mu2, mu1 - mu2 ) + np.trace(sigma1 + sigma2 - 2.0 * covmean.real)
  return FID

###################################################################
# Calculate Frechet Inception Distance (FID)
def get_activation(model, images):
  images = scale_images(images, (299,299,3))
  # pre-process images
  images = preprocess_input(images)
	# calculate activations
  activation = model.predict(images)
  return activation

###################################################################
# Get fake images from a generator
def get_img_fake(generator, nbr_img=1000, latent_dim=100):
  X_latent = tf.random.normal(shape =[nbr_img, latent_dim])
  # Assume generated images are normalized in [-1,1]
  img_fake = generator(X_latent, training = False) * 127.5 + 127.5
  return img_fake

###################################################################
# Show some images
def print_images(images):
  print("Images max", np.max(images), "min", np.min(images), "shape:", images.shape)
  for i in range(8*8):
      plt.subplot(8, 8, i + 1)
      plt.imshow(images[i, :, :, 0], cmap ='gray_r')
      plt.axis('off')
  plt.show()

# Activation of real images from MNIST
def get_act_real(model, nbr_img):
  (img_real, _), (i, _) = tf.keras.datasets.mnist.load_data()
  img_real = img_real[np.random.randint(0, img_real.shape[0], nbr_img)] # select img
  img_real = np.expand_dims(img_real, axis=-1).astype(np.float32)
  print_images(img_real)
  act_real = get_activation(model, img_real)
  return act_real

# Activation of fake images from a generator
def FID_from_generator(filename, model, act_real, nbr_img = 10000):
  # load generator from a folder
  generator = tf.keras.models.load_model(filename)
  #tf.random.set_seed(500)
  img_fake = get_img_fake(generator, nbr_img)
  print_images(img_fake)
  act_fake = get_activation(model, img_fake)

  FID = compute_FID(act_real, act_fake)
  print(f"---> FID: {FID:.5f} for MNIST vs {filename}, {nbr_img} images")
  with open("Result_FID.txt", "a") as file:
    file.write(f"---> FID: {FID:.5f} for MNIST vs {filename}, {nbr_img} images\n")

###################################################################
# Main

# Compute FID on a collection of N images
# "We recommend using a minimum sample size of 10'000 to calculate the FID
# otherwise the true FID of the generator is underestimated."

def runFIDonGenerators(generator_name, nbr_img=10000):
  # Prepare the Inception V3 model
  model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
  act_real = get_act_real(model, nbr_img)
  for g_name in generator_name:
    FID_from_generator(g_name, model, act_real, nbr_img)


generator_name = ['Run1_for_FID/g_model095/',
                  'Run1_for_FID/g_model018/',
                  'Run1_for_FID/g_model003/',
                  'Run1_for_FID/g_model001/']

runFIDonGenerators(generator_name, nbr_img=100)

# FID: 18.44769 MNIST vs MNIST  11.45
# FID: 97.36988 epoch 95        88.06
# FID: 181.77799 epoch 18
# FID: 329.38080 epoch 3
# FID: 414.50330 epoch 1
