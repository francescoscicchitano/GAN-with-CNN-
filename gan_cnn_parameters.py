from keras.optimizers import Adam


random_seed = 101

real_data_shape = (28, 28, 1)
noise_data_shape = 100

n_epochs = 6900
number_of_batches = 1
batch_size = 32
burnin = 2


n_outliers_per_sampling = 25
save_steps = 500

gan_outlier_path = '/home/scicchitano/GCN/data/outliers/outliers'
plot_folder = '/home/scicchitano/GCN/data/image_prova/'
plot_folder_prova_cnn_batch_random = '/home/scicchitano/GCN/data/image_prova_batch/'
gan_outliers_prova_batch_random = '/home/scicchitano/GCN/data/outliers_prova_batch/'

# ####### Discriminator ####### #
discriminator_optimizer = Adam(lr=0.002, beta_1=0.5)

d_filters = [32, 64, 128, 256]
d_kernel_size = 3
d_strides = 2
d_last_strides = 1
d_padding = 'same'

d_momentum = 0.8
d_alpha = 0.2
d_dropout = 0.25
d_last_activation = 'sigmoid'

# ####### Discriminator ####### #
optimizer = Adam(lr=0.0002, beta_1=0.5)

g_first_filter = 128
g_filters = [128, 64]
g_dense_size = 7
g_activation = 'relu'

g_momentum = 0.8
g_kernel_size = 3
g_padding = 'same'
g_channels = 1
g_last_activation = 'tanh'
