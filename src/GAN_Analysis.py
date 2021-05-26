' #################################################################################################### '
' ############################### DCGAN Project code - Jeremy Goumaz ################################# '
' ############################################ May 2021 ############################################## '
' #################################################################################################### '

import numpy as np
import tensorflow as tf
#import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import math
import time
import os, glob, shutil, imageio
import ast
import PIL
from IPython import display
from FID import runFID

rng = np.random.default_rng(seed=None)

' #################################################################################################### '
' ############################################## Debug ############################################### '
' #################################################################################################### '

def print_type(var):
    print("Type:",type(var), end='')
    if type(var) is np.ndarray:
        print(" | Shape:",var.shape,"| dtype =",var.dtype)
    else:
        print("")

' #################################################################################################### '
' ############################################### Data ############################################### '
' #################################################################################################### '

def load_MNIST(mode_0to1 = False):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = np.expand_dims(X_train, axis=-1)
    X_train = X_train.astype(np.float32)
    if mode_0to1:
        X_train = X_train / 255.0 # X = X / 255.0 -> Normalize the images to [0, 1]
    else:
        X_train = (X_train - 127.5) / 127.5  # X = (X - 127.5) / 127.5 -> Normalize the images to [-1, 1]
    return X_train

def load_FASHIONMNIST():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = np.expand_dims(X_train, axis=-1)
    X_train = X_train.astype(np.float32)
    X_train = (X_train - 127.5) / 127.5
    return X_train

def generate_real_samples(n_samples, dataset):
    global rng
    new_index = rng.integers(low=0, high=dataset.shape[0], size=n_samples)
    X_new = dataset[new_index]
    y_new = np.ones((n_samples, 1))
    return X_new, y_new

def generate_fake_samples(n_samples, latent_dim, g_model):
    # Without Generator: newX = rng.uniform(low = -1.0, high = 1.0, size = n_samples*28*28).reshape((n_samples, 28, 28, 1))
    global rng
    latent_vector = generate_latent_vectors(n_samples, latent_dim)
    X_new = g_model.predict(latent_vector)
    y_new = np.zeros((n_samples, 1))
    return X_new, y_new

def generate_latent_vectors(n_samples, latent_dim):
    global rng
    return rng.normal(size=(n_samples, latent_dim))

' #################################################################################################### '
' ############################################## Models ############################################## '
' #################################################################################################### '

def create_discriminator(GANparams):
    model = tf.keras.models.Sequential()

    # layer 1 -> layer "nb_layers"
    for i in range(GANparams["nb_layers"]):
        kwargs = {}
        if i == 0:
            kwargs.update({'input_shape':GANparams["input_shape"]})
        if GANparams["use_initializer_RandomNormal"]:
            kwargs.update({'kernel_initializer':tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)})
        if GANparams["use_Dense"]:
            initial_size = GANparams["input_shape"][0] * GANparams["input_shape"][1] * GANparams["input_shape"][2]
            expo = round(math.log(initial_size) / math.log(2))
            kwargs.update({'units': 2 ** (expo - i)})
            model.add(tf.keras.layers.Dense(**kwargs))
        if GANparams["use_Conv2D"]:
            kwargs.update({'filters': (2**i)*GANparams["Conv2D-filters-small"],
                           'kernel_size': (GANparams["Conv2D-kernel_size"], GANparams["Conv2D-kernel_size"]),
                           'strides': (GANparams["Conv2D-strides"], GANparams["Conv2D-strides"]),
                           'padding': 'same'})
            model.add(tf.keras.layers.Conv2D(**kwargs))
        if GANparams["use_AveragePooling2D"]:
            tf.keras.layers.AveragePooling2D(pool_size=(GANparams["AveragePooling2D-pool_size"], GANparams["AveragePooling2D-pool_size"]))
        if i != 0 and GANparams["use_BatchNormalization"]:
            model.add(tf.keras.layers.BatchNormalization())
        if GANparams["use_ReLU"]:
            model.add(tf.keras.layers.ReLU())
        if GANparams["use_LeakyReLU"]:
            model.add(tf.keras.layers.LeakyReLU(alpha=GANparams["LeakyReLU-alpha"]))
        if GANparams["use_ELU"]:
            model.add(tf.keras.layers.ELU())
        if GANparams["use_Dropout"]:
            model.add(tf.keras.layers.Dropout(GANparams["Dropout-rate"]))

    # last layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation=GANparams["last_layer-activation"]))
    kwargs = {'loss': 'binary_crossentropy',
              'metrics': ['accuracy','binary_accuracy','binary_crossentropy','hinge',tf.keras.metrics.TruePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives()]}
    if GANparams["use_optimizer_Adam"]:
        kwargs.update({'optimizer':tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)})
    model.compile(**kwargs)

    return model

def create_generator(GANparams):
    model = tf.keras.models.Sequential()

    # layer 1 -> layer "nb_layers"
    for i in range(GANparams["nb_layers"]):
        kwargs = {}
        if GANparams["use_initializer_RandomNormal"]:
            kwargs.update({'kernel_initializer':tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)})
        if i == 0:
            kwargs.update({'input_dim': GANparams["latent_dim"]})
        if GANparams["use_Dense"]:
            final_size = GANparams["output_shape"][0] * GANparams["output_shape"][1] * GANparams["output_shape"][2]
            expo = round(math.log(final_size) / math.log(2))
            kwargs.update({'units': 2 ** (expo + i + 1 - GANparams["nb_layers"])})
            model.add(tf.keras.layers.Dense(**kwargs))
        if GANparams["use_Conv2D"] or GANparams["use_Conv2DTranspose"]:
            if i == 0:
                kwargs.update({'units': GANparams["Conv2DTranspose-filters-small"] * GANparams['output_shape'][0] * GANparams['output_shape'][1] / (2**(GANparams["nb_layers"]-1))})
                model.add(tf.keras.layers.Dense(**kwargs))
                model.add(tf.keras.layers.Reshape((int(GANparams['output_shape'][0]/(2**(GANparams["nb_layers"]-1))), int(GANparams['output_shape'][1]/(2**(GANparams["nb_layers"]-1))), (2**(GANparams["nb_layers"]-1)) * GANparams["Conv2DTranspose-filters-small"])))
            else:
                if GANparams["use_UpSampling2D"]:
                    model.add(tf.keras.layers.UpSampling2D(size=(GANparams["UpSampling2D-size"], GANparams["UpSampling2D-size"])))
                if GANparams["use_Conv2D"]:
                    kwargs.update({'filters': (2**(GANparams["nb_layers"]-1-i)) * GANparams["Conv2D-filters-small"],
                                   'kernel_size': (GANparams["Conv2D-kernel_size"], GANparams["Conv2D-kernel_size"]),
                                   'strides': (GANparams["Conv2D-strides"], GANparams["Conv2D-strides"]),
                                   'padding': 'same'})
                    model.add(tf.keras.layers.Conv2D(**kwargs))
                if GANparams["use_Conv2DTranspose"]:
                    kwargs.update({'filters': (2**(GANparams["nb_layers"]-1-i)) * GANparams["Conv2DTranspose-filters-small"],
                                   'kernel_size': (GANparams["Conv2DTranspose-kernel_size"], GANparams["Conv2DTranspose-kernel_size"]),
                                   'strides': (GANparams["Conv2DTranspose-strides"],GANparams["Conv2DTranspose-strides"]),
                                   'padding': 'same'})
                    model.add(tf.keras.layers.Conv2DTranspose(**kwargs))
        if GANparams["use_BatchNormalization"]:
            model.add(tf.keras.layers.BatchNormalization())
        if GANparams["use_ReLU"]:
            model.add(tf.keras.layers.ReLU())
        if GANparams["use_LeakyReLU"]:
            model.add(tf.keras.layers.LeakyReLU(alpha=GANparams["LeakyReLU-alpha"]))
        if GANparams["use_ELU"]:
            model.add(tf.keras.layers.ELU())
        if GANparams["use_Dropout"]:
            model.add(tf.keras.layers.Dropout(GANparams["Dropout-rate"]))

    # last layer
    kwargs = {'activation': GANparams["last_layer-activation"]}
    if GANparams["use_Dense"]:
        kwargs.update({'units': GANparams["output_shape"][0]*GANparams["output_shape"][1]*GANparams["output_shape"][2]})
        model.add(tf.keras.layers.Dense(**kwargs))
        model.add(tf.keras.layers.Reshape((GANparams["output_shape"][0],GANparams["output_shape"][1],GANparams["output_shape"][2])))
    if GANparams["use_Conv2D"]:
        kwargs.update({'filters': 1,
                       'kernel_size': (GANparams["last_layer-kernel_size"], GANparams["last_layer-kernel_size"]),
                       'padding': 'same'
                       })
        model.add(tf.keras.layers.Conv2D(**kwargs))
    if GANparams["use_Conv2DTranspose"]:
        kwargs.update({'filters': 1,
                       'kernel_size': (GANparams["last_layer-kernel_size"], GANparams["last_layer-kernel_size"]),
                       'padding': 'same'
                       })
        model.add(tf.keras.layers.Conv2DTranspose(**kwargs))
    return model

def create_gan(g_model, d_model, opti = True):
    d_model.trainable = False
    model = tf.keras.models.Sequential()
    model.add(g_model)
    model.add(d_model)
    if opti:
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy','binary_accuracy','binary_crossentropy','hinge',tf.keras.metrics.TruePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives()])
    else:
        model.compile(loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy', 'binary_crossentropy', 'hinge', tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    return model

def check_model(GANparams):
    if type(GANparams) is list:
        for GANparam in GANparams:
            d = create_discriminator(GANparam[0])
            d.summary()
            g = create_generator(GANparam[1])
            g.summary()

' #################################################################################################### '
' ############################################# Training ############################################# '
' #################################################################################################### '

def train(g_model, d_model, gan_model, dataset, run_name ="0", n_epochs=100, batch_size=256, latent_dim=100):
    global rng
    n_batches = math.floor(dataset.shape[0] / batch_size)
    dict_d, dict_gan = {}, {}
    time_start = time.time()
    for epoch in range(n_epochs):
        for i in range(n_batches):
            ''' Dataset '''
            X_real, y_real = generate_real_samples(math.floor(batch_size / 2), dataset)
            X_fake, y_fake = generate_fake_samples(math.floor(batch_size / 2), latent_dim, g_model)
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))

            ''' Discriminator training '''
            d_scores = d_model.train_on_batch(X, y)
            dict_d.update(subdict_model(epoch, i, round(time.time()-time_start,2), d_model, d_scores))
            #d_scores_real = d_model.train_on_batch(X_real, y_real)
            #d_scores_fake = d_model.train_on_batch(X_fake, y_fake)
            #dict_d.update(subdict_discriminator(epoch, i, round(time.time() - time_start, 2), d_model, d_scores_real, d_scores_fake))
            #d_loss = (d_scores_real[0] + d_scores_fake[0])/2
            #d_accuracy = (d_scores_real[1] + d_scores_fake[1])/2

            ''' Aversarial training '''
            latent_vectors = generate_latent_vectors(batch_size, latent_dim)
            y_fool = np.ones((batch_size, 1))
            gan_scores = gan_model.train_on_batch(latent_vectors, y_fool)
            dict_gan.update(subdict_model(epoch, i, round(time.time()-time_start,2), gan_model, gan_scores))

            print("Epoch "+str(epoch)+" | "+str(i)+" | "+str(round(time.time()-time_start,2))+" sec | d: "+str(round(d_scores[0],4))+" / "+str(round(d_scores[1],4))+" | gan: "+str(round(gan_scores[0],4))+" / "+str(round(gan_scores[1],4)))
            #print("Epoch "+str(epoch)+" | "+str(i)+" | "+str(round(time.time()-time_start,2))+" sec | d: "+str(round(d_loss,4))+" / "+str(round(d_accuracy,4))+" ("+str(round(d_scores_fake[1],4))+") | gan: "+str(round(gan_scores[0],4))+" / "+str(round(gan_scores[1],4)))

        ''' Save data '''
        save_all(g_model, d_model, gan_model, dict_d, dict_gan, epoch, run_name)
        dict_d.clear()
        dict_gan.clear()

        X_fake, y_fake = generate_fake_samples(100, 100, g_model)
        save_image(X_fake, epoch, run_name)

' #################################################################################################### '
' ############################################# Results ############################################## '
' #################################################################################################### '

def subdict_model(epoch, i, time, model, scores):
    dict_metrics = {}
    for j in range(len(scores)):
        dict_metrics.update({model.metrics_names[j]: scores[j]})
    return {(epoch+1, i+1, time): dict_metrics}

def subdict_discriminator(epoch, i, time, model, scores_real, scores_fake):
    dict_metrics = {}
    for j in range(len(scores_real)):
        dict_metrics.update({"real_"+model.metrics_names[j]: scores_real[j]})
    for j in range(len(scores_fake)):
        dict_metrics.update({"fake_"+model.metrics_names[j]: scores_fake[j]})
    return {(epoch+1, i+1, time): dict_metrics}

def save_all(g_model, d_model, gan_model, dict_d, dict_gan, epoch, name):
    g_model.save("Run_"+str(name)+"/g_model{epoch:03d}".format(epoch = epoch+1))
    if epoch%10 == 9: # it can be changed
        d_model.save("Run_"+str(name)+"/d_model{epoch:03d}".format(epoch=epoch + 1))
        gan_model.save("Run_"+str(name)+"/gan_model{epoch:03d}".format(epoch=epoch + 1))
    with open(file="Run_"+str(name)+"/d_model_scores.txt", mode='a') as f:
        f.write(str(dict_d)+"\n")
    with open(file="Run_"+str(name)+"/gan_model_scores.txt", mode='a') as f:
        f.write(str(dict_gan)+"\n")

def save_image(images, epoch=None, run_name=None, index=None, destination_folder=None):
    nb_images = images.shape[0]
    n = math.ceil(math.sqrt(nb_images))

    final_image = np.full((30 * n - 2, 30 * n - 2), np.min(images))
    for i in range(nb_images):
        final_image[30 * (i // n):30 * (i // n) + 28, 30 * (i % n):30 * (i % n) + 28] = images[i, :, :, 0]
    plt.imshow(final_image, cmap='gray_r')
    plt.axis('off')

    fig_name = ""
    if run_name is not None:
        fig_name += "Run_" + str(run_name) + "/"
        if os.path.isdir(fig_name) is False:
            os.mkdir(fig_name)
    if destination_folder is not None:
        fig_name += str(destination_folder)
        if destination_folder[-1] != "/":
            fig_name += "/"
        if os.path.isdir(fig_name) is False:
            os.mkdir(fig_name)
    fig_name += "generated_plot"
    if epoch is not None:
        fig_name += "_e{epoch:03d}".format(epoch = epoch + 1)
    if index is not None:
        fig_name += "_{index:03d}".format(index = index + 1)
    fig_name += ".png"

    plt.savefig(fig_name, dpi=150)
    plt.close()

def save_parameters_used(GANparams_d, GANparams_g, run_name=""):
    if os.path.isdir("Run_" + run_name) is False: os.mkdir("Run_" + run_name)
    with open(file="Run_" + run_name + "/args_used.txt", mode='w') as f:
        f.write("########## GANparams for discriminator:\n")
        for key, val in GANparams_d.items():
            f.write(str(key) + ": " + str(val) + "\n")
        f.write("\n########## GANparams for generator:\n")
        for key, val in GANparams_g.items():
            f.write(str(key) + ": " + str(val) + "\n")

' #################################################################################################### '
' ############################################ Analysis ############################################# '
' #################################################################################################### '

def create_gif_from_folder(source_folder=None, extension="png", slow_start=False):
   if source_folder is None:
       path = ""
   else:
       path = source_folder
       if path[-1] != "/":
            path += "/"
   images = {}
   for name in glob.glob(path + "*." + extension):
       char_admitted = "0123456789"
       index = int("".join([char for char in name if char in char_admitted]))
       images.update({index:name})
   images = list(images.items())
   images.sort(key=lambda x: x[0])
   with imageio.get_writer(path+"evolution.gif", mode='I', fps=30) as writer:
       for i in range(len(images)):
           image_to_add = imageio.imread(images[i][1])
           writer.append_data(image_to_add)
           if slow_start:
               ind = [30] # 30 is a parameter that can be changed
               for j in range(7): # 7 is a parameter that can be changed
                   ind.append(math.floor(ind[j]/1.5)) # 1.5 is a parameter that can be changed
                   if i<ind[j]:
                       writer.append_data(image_to_add)

def create_gif_from_run(run_folder=None, nb_gif=1, destination_folder=None):
    time_start = time.time()
    if run_folder is None:
        path = ""
    else:
        path = run_folder
        if path[-1] != "/":
            path += "/"
    models = {}
    for name in glob.glob(path + "g_model*/"):
        char_admitted = "0123456789"
        index = int("".join([char for char in name if char in char_admitted]))
        models.update({index : name[0:-1]})
    models = list(models.items())
    models.sort(key=lambda x: x[0])
    folder_gif = path + "gif_generation_tmp"
    if os.path.isdir(folder_gif):
        shutil.rmtree(folder_gif)
    os.mkdir(folder_gif)
    for n in range(nb_gif):
        os.mkdir(folder_gif + "/" + str(n + 1))
    latent_vectors = generate_latent_vectors(100 * nb_gif, 100)
    g_model = tf.keras.models.load_model(models[0][1])
    for i in range(len(models)):
        print("Gif creation:",i,"/",len(models),"|",round(time.time()-time_start,2),"sec")
        g_model.load_weights(models[i][1])
        X_predicted = g_model.predict(latent_vectors)
        for n in range(nb_gif):
            image = np.full((298,298),np.min(X_predicted))
            for j in range(100):
                image[30*(j//10):30*(j//10)+28,30*(j%10):30*(j%10)+28] = X_predicted[j + n * 100, :, :, 0]
            plt.imshow(image, cmap='gray_r')
            plt.axis('off')
            fig_name = folder_gif + "/" + str(n+1) + "/generated_plot_{index:03d}.png".format(index= i + 1)
            plt.savefig(fig_name, dpi=150)
            plt.close()
    for n in range(nb_gif):
        create_gif_from_folder(folder_gif + "/" + str(n+1), slow_start=True)
        if destination_folder is None:
            shutil.copyfile(folder_gif + "/" + str(n+1) + "/evolution.gif", path+"evolution{n:02d}.gif".format(n = n + 1))
        else:
            shutil.copyfile(folder_gif + "/" + str(n + 1) + "/evolution.gif", destination_folder + "/evolution{n:02d}.gif".format(n=n + 1))
    shutil.rmtree(folder_gif)

def create_images_from_model(g_model, nb_im=1, destination_folder = None):
   latent_vectors = generate_latent_vectors(100 * nb_im, 100)
   X_predicted = g_model.predict(latent_vectors)
   for n in range(nb_im):
       save_image(images = X_predicted[100*n:100*n+100], index = n, destination_folder = destination_folder)

def load_scores(filename):
   if filename[-4:] != ".txt":
       filename = filename + ".txt"
   scores = {}
   with open(filename, mode='r') as f:
       if f.readable():
           for line in f:
               part_scores = ast.literal_eval(line)
               if type(part_scores) is dict:
                   scores.update(ast.literal_eval(line))
               else:
                   print("Problem during \"load_scores\" !")
                   return scores
   return scores

def plot_score(filename, metrics_given = ['accuracy'], destination_folder = None, window_mode = False, moving_average_bilateral = True):
    ''' metric = 'loss', 'accuracy', 'hinge', 'true_positives', 'true_negatives', 'false_positives', 'false_negatives' with prefix real_ or fake_ for discriminator '''
    scores = load_scores(filename)
    scores = list(scores.items())
    existing_metrics = list(scores[0][1].keys())

    metrics_to_compute = ['accuracy','loss'] # for the discriminator when we only have real_accuracy and fake_accuracy for example -> compute accuracy = (real_accuracy + fake_accuracy)/2
    for m in existing_metrics:
        if m[0:5] == 'real_':
            metric_name = m[5:]
            if metric_name in metrics_to_compute:
                for i in range(len(scores)):
                    value = (scores[i][1]['real_'+metric_name] + scores[i][1]['fake_'+metric_name])/2
                    scores[i][1].update({metric_name:value})
                existing_metrics.append(metric_name)

    for metric in metrics_given:
        if metric.lower() in existing_metrics:
            n_epoch, n_batches, runtime = scores[-1][0]
            x, y = [], []
            for i in range(len(scores)):
                x.append(scores[i][0][0]-1+(scores[i][0][1]-1)/n_batches)
                y.append(scores[i][1][metric])
            if moving_average_bilateral:
                window = 234
                y_smooth = np.convolve(y, np.ones(window), 'same') / np.convolve(np.ones(len(y)), np.ones(window), 'same')
            plt.clf()
            plt.plot(x, y, x, y_smooth)
            plt.xlabel("Epoch")
            metric_title = metric[0].upper()+metric[1:].lower()
            plt.ylabel(metric_title)
            title = ""
            filename_txt = filename[filename.rfind("/")+1:]
            if filename_txt[0] == 'd':
                title = "Discriminator Model"
            if filename_txt[0] == 'g':
                title = "GAN Model"
            plt.title(title)
            plt.legend(["All data points","Moving Average (1 epoch)"])
            plt.margins(x=1e-4)
            if window_mode:
                plt.show()
            if destination_folder is not None:
                fig_name = destination_folder + "/" + title + " - " + metric_title
                plt.savefig(fig_name, dpi=150)
            plt.clf()
    return n_epoch, n_batches, runtime

def create_FID_results(g_model, model_filename, results_folder, n_img=20000, n_avg=2):
    FID_results = runFID(g_model, nbr_img=n_img, nbr_avg=n_avg, filename=model_filename) # {"FIDs": , "FID_mean": , "n_img": , "n_avg": , "model_filename": , "time_fid": , "time_run": }
    FIDs = FID_results['FIDs']
    FIDs_round = []
    for fid in FIDs:
        FIDs_round.append(round(fid,5))
    FID_results_str = f"---> FID: {round(FID_results['FID_mean'],5)} for MNIST vs {FID_results['model_filename']}, {FID_results['n_img']} images (averaged from {FID_results['n_avg']} computations: {FIDs_round})\n"
    timing = f"     Initialization of the run: {round(FID_results['time_run'][0],2)} sec\n"
    for i in range(len(FID_results['FIDs'])):
        if i == 0:
            computation_time = round(FID_results['time_fid'][i],2)
        else:
            computation_time = round(FID_results['time_fid'][i]-FID_results['time_fid'][i-1], 2)
        timing += f"       FID computation {i}: {computation_time} sec (FID = {round(FID_results['FIDs'][i],5)})\n"
    timing += f"     End of the run: {round(FID_results['time_run'][1],2)} sec\n\n"

    with open(results_folder+"/FID_results.txt",mode='a') as f:
        f.write(FID_results_str)
        f.write(timing)

def create_results(run_folder = None):
    t1 = time.time()
    run_name = None
    if run_folder is None:
        results_folder = "Run_results"
        g_model_folders = "g_model*/"
        d_scores_file = "d_model_scores.txt"
        gan_scores_file = "gan_model_scores.txt"
    else:
        if run_folder[-1] == "/":
            run_folder = run_folder[0:-1]
        l = run_folder.rfind("/")
        if run_folder[l + 1:l + 5] == "Run_":
            run_name = run_folder[l + 5:]
        results_folder = run_folder + "_results"
        g_model_folders = run_folder + "/g_model*/"
        d_scores_file = run_folder + "/d_model_scores.txt"
        gan_scores_file = run_folder + "/gan_model_scores.txt"
    if os.path.isdir(results_folder) is False: os.mkdir(results_folder)
    t2 = time.time()

    ''' gif creation '''
    if os.path.isdir(results_folder+"/gif") is False: os.mkdir(results_folder+"/gif")
    create_gif_from_run(run_folder=run_folder, nb_gif=1, destination_folder=results_folder+"/gif")
    t3 = time.time()

    ''' images creation (from last g_model) '''
    models = {}
    for name in glob.glob(g_model_folders):
        char_admitted = "0123456789"
        index = int("".join([char for char in name if char in char_admitted]))
        models.update({index : name[0:-1]})
    models = list(models.items())
    models.sort(key=lambda x: x[0])
    g_model_final = tf.keras.models.load_model(models[-1][1])
    if os.path.isdir(results_folder + "/images_last_epoch") is False: os.mkdir(results_folder + "/images_last_epoch")
    create_images_from_model(g_model_final, nb_im=20, destination_folder=results_folder + "/images_last_epoch")
    t4 = time.time()

    ''' plots creation '''
    n_epoch, n_batches, train_runtime = plot_score(d_scores_file, metrics_given=['accuracy','loss','real_accuracy','fake_accuracy','real_loss','fake_loss'], destination_folder=results_folder)
    plot_score(gan_scores_file, metrics_given=['accuracy','loss'], destination_folder=results_folder)
    t5 = time.time()

    d_model = tf.keras.models.load_model(run_folder + "/d_model010")
    g_model = tf.keras.models.load_model(run_folder + "/g_model010")
    with open(results_folder + "/model_discriminator.txt", mode='w') as f:
        d_model.summary(print_fn=lambda x: f.write(x+"\n"))
    with open(results_folder + "/model_generator.txt", mode='w') as f:
        g_model.summary(print_fn=lambda x: f.write(x+"\n"))

    create_FID_results(g_model=g_model_final, model_filename=models[-1][1], results_folder=results_folder, n_img=20000, n_avg=2)
    t6 = time.time()

    with open(results_folder+"/results_computation.txt",mode='w') as f:
        f.write("Results creation runtime:\n")
        f.write(f"  Initialization: {round(t2 - t1,2)} sec\n")
        f.write(f"  Gif creation: {round(t3 - t2, 2)} sec\n")
        f.write(f"  Images creation: {round(t4 - t3, 2)} sec\n")
        f.write(f"  Plots creation: {round(t5 - t4, 2)} sec\n")
        f.write(f"  FID computation: {round(t6 - t5, 2)} sec\n")
        f.write(f"Total runtime: {round(t6 - t1, 2)} sec\n\n")
        f.write("Architecture:\n")
        f.write(f"  Run name: {run_name}\n")
        f.write(f"  Run folder: {run_folder}\n")
        f.write(f"  Results folder: {results_folder}\n")
        f.write(f"  Generator model used (final): {models[-1][1]}\n\n")
        f.write(f"Training:\n")
        f.write(f"  Number of epochs: {n_epoch}\n")
        f.write(f"  Number of batches per epoch: {n_batches}\n")
        f.write(f"  Number of images per batch: 256\n")
        f.write(f"  Training runtime: {round(train_runtime,2)} sec\n\n")

        d_args_used = []
        g_args_used = []
        if os.path.isfile(run_folder + "/args_used.txt"):
            with open(run_folder + "/args_used.txt", mode='r') as f2:
                if f2.readable():
                    i = 0
                    for line in f2:
                        if line[0] == "#": i += 1
                        if line[0] != " " and line[0] != "#" and line[0] != "\n":
                            val = line.split(':')
                            val[0] = val[0].strip()
                            val[1] = val[1].strip()
                            if i == 1:
                                d_args_used.append(val)
                            if i == 2:
                                g_args_used.append(val)

        d_args = []
        g_args = []
        for key, val in GANparams_d_default.items():
            d_args.append([str(key), str(val)])
        for key, val in GANparams_g_default.items():
            g_args.append([str(key), str(val)])

        d_args_changed = []
        for arg in d_args_used:
            changed = True
            for arg2 in d_args:
                if arg == arg2:
                    changed = False
            if changed:
                d_args_changed.append(arg)
        g_args_changed = []
        for arg in g_args_used:
            changed = True
            for arg2 in g_args:
                if arg == arg2:
                    changed = False
            if changed:
                g_args_changed.append(arg)

        f.write(f"Test done:\n")
        f.write(f"  ########## GANparams changes in this test for discriminator (compared to GANparams_default):\n")
        for arg in d_args_changed:
            f.write(f"  {arg[0]}: {arg[1]}\n")
        if len(d_args_changed) == 0:
            f.write(f"  None\n")
        f.write(f"  ########## GANparams changes in this test for generator (compared to GANparams_default):\n")
        for arg in g_args_changed:
            f.write(f"  {arg[0]}: {arg[1]}\n")
        if len(g_args_changed) == 0:
            f.write(f"  None\n")

        f.write(f"\nModels used:\n")
        if os.path.isfile(run_folder+"/args_used.txt"):
            with open(run_folder+"/args_used.txt",mode='r') as f2:
                if f2.readable():
                    for line in f2:
                        f.write(f"  {line}")
        f.write(f"\n  ########## params for GAN model:")
        f.write(f"  discriminator.trainable: False\n")
        f.write(f"  use_optimizer_Adam: True\n")
        f.write(f"  ########## loss function used (in discriminator and GAN models): binary_crossentropy\n\n")
        f.write(f"Dataset: MNIST (digits) 60'0000 images\n")


' #################################################################################################### '
' ############################################### Run ################################################ '
' #################################################################################################### '

GANparams_d_default = {

    # Choice between Dense and Conv2D [1 True only]
    "use_Dense": False,  # use Dense in each layer
    "use_Conv2D": True,  # use Conv2D in each layer (except the last one which use Dense)

    # If Conv2D, AveragePooling can be activated -> Conv2D-strides should be decreased to 1
    "Conv2D-strides": 2,  # stride parameter for Conv2D
    "use_AveragePooling2D": False,  # use AveragePooling in each layer (except the last one)

    # Additional layers
    "use_BatchNormalization": True,  # use batch normalization in each layer except the first and last one
    "use_ReLU": False,  # use ReLU activation after each layer except the last one
    "use_LeakyReLU": True,  # use LeakyReLU activation after each layer except the last one
    "use_ELU": False,
    "use_Dropout": True,  # use Dropout after each layer except the last one
    "last_layer-activation": 'sigmoid',  # last activation function used (in the last layer)

    # These parameters should not be changed for the tests
    "input_shape": (28, 28, 1),  # (28, 28, 1) for MNIST digits
    "use_initializer_RandomNormal": False,  # use normal weight initialization in each layer
    "nb_layers": 2,  # no max value but increasing this parameter has a very high computational cost / and the discriminator should not be too good
    "Conv2D-filters-small": 64,  # filters parameter for Conv2D
    "Conv2D-kernel_size": 3,  # kernel_size parameter for Conv2D
    "AveragePooling2D-pool_size": 2,  # pool_size for AveragePooling2D
    "LeakyReLU-alpha": 0.2,  # alpha parameter for LeakyReLU
    "Dropout-rate": 0.3,  # rate parameter for Dropout
    "use_optimizer_Adam": True,  # use Adam optimizer
}

GANparams_g_default = {

    # Choice between Dense and Conv2D and Conv2DTranspose [1 True only]
    "use_Dense": False,  # use Dense in each layer
    "use_Conv2D": False,  # use Conv2D in each layer
    "use_Conv2DTranspose": True,  # use Conv2DTranspose in each layer

    # If Conv2D is chosen: UpSampling2D MUST be activated
    "use_UpSampling2D": False,  # use UpSampling2D in each layer (except the last one)

    # Additional layers
    "use_BatchNormalization": True,  # use batch normalization in each layer except the last one
    "use_ReLU": True,  # use ReLU activation after each layer except the last one
    "use_LeakyReLU": False,  # use LeakyReLU activation after each layer except the last one
    "use_ELU": False,
    "use_Dropout": False,  # use Dropout after each layer except the last one
    "last_layer-activation": 'tanh',  # last activation function used (in the last layer)

    # These parameters should not be changed for the tests
    "latent_dim": 100,  # latent dimension, set to 100
    "use_initializer_RandomNormal": False,  # use normal weight initialization in each layer
    "nb_layers": 3,  # max value: 3 for images 28x28 (because 28 can be divided 2x by 2)
    "Conv2D-filters-small": 64,  # filters parameter for Conv2D
    "Conv2D-kernel_size": 3,  # kernel_size parameter for Conv2D
    "Conv2D-strides": 1,  # stride parameter for Conv2D
    "Conv2DTranspose-filters-small": 64,  # filters parameter for Conv2DTranspose
    "Conv2DTranspose-kernel_size": 3,  # kernel_size parameter for Conv2DTranspose
    "Conv2DTranspose-strides": 2,  # stride parameter for Conv2DTranspose
    "UpSampling2D-size": 2,  # size parameter for UpSampling2D
    "LeakyReLU-alpha": 0.2,  # alpha parameter for LeakyReLU
    "Dropout-rate": 0.3,  # rate parameter for Dropout
    "last_layer-kernel_size": 5,  # kernel_size parameter for Conv2DTranspose (only for the last layer)
    "output_shape": (28, 28, 1),  # (28, 28, 1) for MNIST digits
}

def run_test(GANparams):
    for i in range(len(GANparams)):
        run_name = "test9_"+str(GANparams[i][2])
        save_parameters_used(GANparams[i][0], GANparams[i][1], run_name)
        d_model = create_discriminator(GANparams[i][0])
        g_model = create_generator(GANparams[i][1])
        gan_model = create_gan(g_model, d_model, opti = True)
        dataset = load_MNIST()
        train(g_model, d_model, gan_model, dataset, run_name, n_epochs=100)

' #################################################################################################### '
' ############################################## Main ################################################ '
' #################################################################################################### '

GANparams_d1 = GANparams_d_default.copy()
GANparams_g1 = GANparams_g_default.copy()
GANparams_g1.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
T1 = (GANparams_d1, GANparams_g1, 1)

GANparams_d2 = GANparams_d_default.copy()
GANparams_g2 = GANparams_g_default.copy()
GANparams_g2.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_d2.update({"use_ReLU":True, "use_LeakyReLU":False})
T2 = (GANparams_d2, GANparams_g2, 2)

GANparams_d3 = GANparams_d_default.copy()
GANparams_g3 = GANparams_g_default.copy()
GANparams_g3.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_d3.update({"use_BatchNormalization":False})
T3 = (GANparams_d3, GANparams_g3, 3)

GANparams_d4 = GANparams_d_default.copy()
GANparams_g4 = GANparams_g_default.copy()
GANparams_g4.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_g4.update({"use_BatchNormalization":False})
T4 = (GANparams_d4, GANparams_g4, 4)

GANparams_d5 = GANparams_d_default.copy()
GANparams_g5 = GANparams_g_default.copy()
GANparams_g5.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_d5.update({"use_BatchNormalization":False})
GANparams_g5.update({"use_BatchNormalization":False})
T5 = (GANparams_d5, GANparams_g5, 5)

GANparams_d6 = GANparams_d_default.copy()
GANparams_g6 = GANparams_g_default.copy()
GANparams_g6.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_d6.update({"use_Dropout":False})
T6 = (GANparams_d6, GANparams_g6, 6)

GANparams_d7 = GANparams_d_default.copy()
GANparams_g7 = GANparams_g_default.copy()
GANparams_g7.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_g7.update({"use_Dropout":True})
T7 = (GANparams_d7, GANparams_g7, 7)



GANparams_d8 = GANparams_d_default.copy()
GANparams_g8 = GANparams_g_default.copy()
GANparams_d8.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
T8 = (GANparams_d8, GANparams_g8, 8)

GANparams_d9 = GANparams_d_default.copy()
GANparams_g9 = GANparams_g_default.copy()
GANparams_d9.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
GANparams_d9.update({"use_ReLU":True, "use_LeakyReLU":False})
T9 = (GANparams_d9, GANparams_g9, 9)

GANparams_d10 = GANparams_d_default.copy()
GANparams_g10 = GANparams_g_default.copy()
GANparams_d10.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
GANparams_d10.update({"use_BatchNormalization":False})
T10 = (GANparams_d10, GANparams_g10, 10)

GANparams_d11 = GANparams_d_default.copy()
GANparams_g11 = GANparams_g_default.copy()
GANparams_d11.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
GANparams_g11.update({"use_BatchNormalization":False})
T11 = (GANparams_d11, GANparams_g11, 11)

GANparams_d12 = GANparams_d_default.copy()
GANparams_g12 = GANparams_g_default.copy()
GANparams_d12.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
GANparams_d12.update({"use_BatchNormalization":False})
GANparams_g12.update({"use_BatchNormalization":False})
T12 = (GANparams_d12, GANparams_g12, 12)

GANparams_d13 = GANparams_d_default.copy()
GANparams_g13 = GANparams_g_default.copy()
GANparams_d13.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
GANparams_d13.update({"use_Dropout":False})
T13 = (GANparams_d13, GANparams_g13, 13)

GANparams_d14 = GANparams_d_default.copy()
GANparams_g14 = GANparams_g_default.copy()
GANparams_d14.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
GANparams_g14.update({"use_Dropout":True})
T14 = (GANparams_d14, GANparams_g14, 14)



GANparams_d15 = GANparams_d_default.copy()
GANparams_g15 = GANparams_g_default.copy()
GANparams_g15.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_d15.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
T15 = (GANparams_d15, GANparams_g15, 15)

GANparams_d16 = GANparams_d_default.copy()
GANparams_g16 = GANparams_g_default.copy()
GANparams_g16.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_d16.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
GANparams_d16.update({"use_ReLU":True, "use_LeakyReLU":False})
T16 = (GANparams_d16, GANparams_g16, 16)

GANparams_d17 = GANparams_d_default.copy()
GANparams_g17 = GANparams_g_default.copy()
GANparams_g17.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_d17.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
GANparams_d17.update({"use_BatchNormalization":False})
T17 = (GANparams_d17, GANparams_g17, 17)

GANparams_d18 = GANparams_d_default.copy()
GANparams_g18 = GANparams_g_default.copy()
GANparams_g18.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_d18.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
GANparams_g18.update({"use_BatchNormalization":False})
T18 = (GANparams_d18, GANparams_g18, 18)

GANparams_d19 = GANparams_d_default.copy()
GANparams_g19 = GANparams_g_default.copy()
GANparams_g19.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_d19.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
GANparams_d19.update({"use_BatchNormalization":False})
GANparams_g19.update({"use_BatchNormalization":False})
T19 = (GANparams_d19, GANparams_g19, 19)

GANparams_d20 = GANparams_d_default.copy()
GANparams_g20 = GANparams_g_default.copy()
GANparams_g20.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_d20.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
GANparams_d20.update({"use_Dropout":False})
T20 = (GANparams_d20, GANparams_g20, 20)

GANparams_d21 = GANparams_d_default.copy()
GANparams_g21 = GANparams_g_default.copy()
GANparams_g21.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_d21.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
GANparams_g21.update({"use_Dropout":True})
T21 = (GANparams_d21, GANparams_g21, 21)



GANparams_d22 = GANparams_d_default.copy()
GANparams_g22 = GANparams_g_default.copy()
GANparams_d22.update({"use_Dense":True, "use_Conv2D":False})
GANparams_g22.update({"use_Dense":True, "use_Conv2DTranspose": False})
T22 = (GANparams_d22, GANparams_g22, 22)

GANparams_d23 = GANparams_d_default.copy()
GANparams_g23 = GANparams_g_default.copy()
GANparams_d23.update({"use_Dense":True, "use_Conv2D":False})
GANparams_g23.update({"use_Dense":True, "use_Conv2DTranspose": False})
GANparams_d23.update({"use_ReLU":True, "use_LeakyReLU":False})
T23 = (GANparams_d23, GANparams_g23, 23)

GANparams_d24 = GANparams_d_default.copy()
GANparams_g24 = GANparams_g_default.copy()
GANparams_d24.update({"use_Dense":True, "use_Conv2D":False})
GANparams_g24.update({"use_Dense":True, "use_Conv2DTranspose": False})
GANparams_d24.update({"use_BatchNormalization":False})
T24 = (GANparams_d24, GANparams_g24, 24)

GANparams_d25 = GANparams_d_default.copy()
GANparams_g25 = GANparams_g_default.copy()
GANparams_d25.update({"use_Dense":True, "use_Conv2D":False})
GANparams_g25.update({"use_Dense":True, "use_Conv2DTranspose": False})
GANparams_g25.update({"use_BatchNormalization":False})
T25 = (GANparams_d25, GANparams_g25, 25)

GANparams_d26 = GANparams_d_default.copy()
GANparams_g26 = GANparams_g_default.copy()
GANparams_d26.update({"use_Dense":True, "use_Conv2D":False})
GANparams_g26.update({"use_Dense":True, "use_Conv2DTranspose": False})
GANparams_d26.update({"use_BatchNormalization":False})
GANparams_g26.update({"use_BatchNormalization":False})
T26 = (GANparams_d26, GANparams_g26, 26)

GANparams_d27 = GANparams_d_default.copy()
GANparams_g27 = GANparams_g_default.copy()
GANparams_d27.update({"use_Dense":True, "use_Conv2D":False})
GANparams_g27.update({"use_Dense":True, "use_Conv2DTranspose": False})
GANparams_d27.update({"use_Dropout":False})
T27 = (GANparams_d27, GANparams_g27, 27)

GANparams_d28 = GANparams_d_default.copy()
GANparams_g28 = GANparams_g_default.copy()
GANparams_d28.update({"use_Dense":True, "use_Conv2D":False})
GANparams_g28.update({"use_Dense":True, "use_Conv2DTranspose": False})
GANparams_g28.update({"use_Dropout":True})
T28 = (GANparams_d28, GANparams_g28, 28)



GANparams_d29 = GANparams_d_default.copy()
GANparams_g29 = GANparams_g_default.copy()
GANparams_d29.update({"use_initializer_RandomNormal":True})
T29 = (GANparams_d29, GANparams_g29, 29)

GANparams_d30 = GANparams_d_default.copy()
GANparams_g30 = GANparams_g_default.copy()
GANparams_g30.update({"use_initializer_RandomNormal":True})
T30 = (GANparams_d30, GANparams_g30, 30)

GANparams_d31 = GANparams_d_default.copy()
GANparams_g31 = GANparams_g_default.copy()
GANparams_d31.update({"use_initializer_RandomNormal":True})
GANparams_g31.update({"use_initializer_RandomNormal":True})
T31 = (GANparams_d31, GANparams_g31, 31)

GANparams_d32 = GANparams_d_default.copy()
GANparams_g32 = GANparams_g_default.copy()
GANparams_d32.update({"use_optimizer_Adam":False})
T32 = (GANparams_d32, GANparams_g32, 32)


GANparams_d33 = GANparams_d_default.copy()
GANparams_g33 = GANparams_g_default.copy()
GANparams_d33.update({"nb_layers":3})
T33 = (GANparams_d33, GANparams_g33, 33)

GANparams_d34 = GANparams_d_default.copy()
GANparams_g34 = GANparams_g_default.copy()
GANparams_g34.update({"use_Conv2DTranspose":False, "use_Conv2D":True, "use_UpSampling2D":True})
GANparams_d34.update({"Conv2D-strides":1, "use_AveragePooling2D":True})
GANparams_d34.update({"nb_layers":3})
T34 = (GANparams_d34, GANparams_g34, 34)

GANparams_d35 = GANparams_d_default.copy()
GANparams_g35 = GANparams_g_default.copy()
GANparams_d35.update({"use_Dense":True, "use_Conv2D":False})
GANparams_g35.update({"use_Dense":True, "use_Conv2DTranspose": False})
GANparams_d35.update({"nb_layers":3})
T35 = (GANparams_d35, GANparams_g35, 35)



set1 = [T1, T8, T15, T22] # default
set2 = [T33, T34, T35] # 3 layers
set3 = [T5, T12, T19, T26] # -2BN
set4 = [T7, T14, T21, T28] # G+DO
set5 = [T2, T9, T16, T23] # D+RELU
set6 = [T29, T30, T31, T32] # initializer + optimizer
set7 = [T3, T4, T10, T11, T17, T18, T24, T25] # -BN
set8 = [T6, T13, T20, T27] # D-DO



check_model(set1)
check_model(set2)
check_model(set3)
check_model(set4)
check_model(set5)
check_model(set6)
check_model(set7)
check_model(set8)



# set1 + set2 here
run_list = ["Runs/GAN_test9_set1/Run_test9_1",
            "Runs/GAN_test9_set1/Run_test9_8",
            "Runs/GAN_test9_set1/Run_test9_15",
            "Runs/GAN_test9_set2/Run_test9_33",
            "Runs/GAN_test9_set2/Run_test9_34"
            ]

for run in run_list:
   create_results(run)







