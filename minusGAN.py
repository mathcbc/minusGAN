from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.data_gmm import GMM_distribution, sample_GMM
from utils.data_utils import shuffle, iter_data
from tqdm import tqdm

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace


""" parameters """
n_epoch = 100
batch_size  = 128
dataset_size_x = 512*4
dataset_size_z = 512*4
dataset_size_y = 512*4

dataset_size_x_test = 512*2
dataset_size_z_test = 512*2
dataset_size_y_test = 512*2

input_dim = 2
latent_dim = 2
eps_dim = 2


n_layer_disc = 2
n_hidden_disc = 256
n_layer_gen = 2
n_hidden_gen= 256
n_layer_inf = 2
n_hidden_inf= 256

learning_rate = 0.001

""" Create directory for results """
result_dir = 'results/'
directory = result_dir
if not os.path.exists(directory):
    os.makedirs(directory)


""" Create dataset """

# create X train dataset
means_x = map(lambda x:  np.array(x), [[1, 1]])
means_x = list(means_x)
std_x = 1
variances_x = [np.eye(2) * std_x for _ in means_x]
priors_x = [1.0/len(means_x) for _ in means_x]

gaussian_mixture = GMM_distribution(means=means_x, variances=variances_x, priors=tuple(priors_x))
dataset_x = sample_GMM(dataset_size_x, means_x, variances_x, priors_x, sources=('features', ))

# plot x
X_dataset  = dataset_x.data['samples']
X_targets = dataset_x.data['label']

fig_mx, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(X_dataset[:, 0], X_dataset[:, 1], c='b',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.axis('on')
save_path_x = result_dir + 'X_gmm_data_train.pdf'
plt.savefig(save_path_x, transparent=True, bbox_inches='tight')


# create Z train dataset
means_z = map(lambda x:  np.array(x), [[0, 0]])
means_z = list(means_z)
std_z = 1.0
variances_z = [np.eye(2) * std_z for _ in means_z]
priors_z = [1.0/len(means_z) for _ in means_z]

dataset_z = sample_GMM(dataset_size_z, means_z, variances_z, priors_z, sources=('features', ))
save_path_z = result_dir + 'Z_gmm_data_train.pdf'

# plot z
Z_dataset = dataset_z.data['samples']
Z_labels = dataset_z.data['label']

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(Z_dataset[:, 0], Z_dataset[:, 1], c='r',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$z_1$'); ax.set_ylabel('$z_2$')
ax.axis('on')
plt.savefig(save_path_z, transparent=True, bbox_inches='tight')

# create Y train dataset
means_y = map(lambda x:  np.array(x), [[-0.5, -0.5]])
means_y = list(means_y)
std_y = 1
variances_y = [np.eye(2) * std_y for _ in means_y]
priors_y = [1.0/len(means_y) for _ in means_y]

dataset_y = sample_GMM(dataset_size_y, means_y, variances_y, priors_y, sources=('features', ))
save_path_y = result_dir + 'Y_gmm_data_train.pdf'

# plot y
Y_dataset = dataset_y.data['samples']
Y_labels = dataset_y.data['label']

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(Y_dataset[:, 0], Y_dataset[:, 1], c='r',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$')
ax.axis('on')
plt.savefig(save_path_y, transparent=True, bbox_inches='tight')

# plot x and y
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(X_dataset[:, 0], X_dataset[:, 1], c='b',
           edgecolor='none', alpha=0.5)
ax.scatter(Y_dataset[:, 0], Y_dataset[:, 1], c='r',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$x_and_y$')
ax.axis('on')
save_path_xy = result_dir + 'XY_gmm_data_train.pdf'
plt.savefig(save_path_xy, transparent=True, bbox_inches='tight')

""" Networks """


def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(mu=tf.zeros(shape), diag_stdev=tf.ones(shape), **kwargs)),  tf.float32)


def generative_network(z, input_dim, n_layer, n_hidden, eps_dim):
    with tf.variable_scope("generative"):
        eps = standard_normal([z.get_shape().as_list()[0], eps_dim], name="eps") * 1.0
        h = tf.concat([z, eps], 1)
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, input_dim, activation_fn=None, scope="p_x")
    return x


def inference_network(x, latent_dim, n_layer, n_hidden, eps_dim):
    with tf.variable_scope("inference"):
        eps = standard_normal([x.get_shape().as_list()[0], eps_dim], name="eps") * 1.0
        h = tf.concat([x, eps], 1)
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        z = slim.fully_connected(h, latent_dim, activation_fn=None, scope="q_z")
    return z


def data_network(x,z, n_layers=2, n_hidden=256, activation_fn=None):
    """Approximate x log data density."""
    h = tf.concat([x,z], 1)
    with tf.variable_scope('discriminator'):
        h = slim.repeat(h, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d, squeeze_dims=[1])


def xy_discriminator(x, n_layers=2, n_hidden=256, activation_fn=None):
    """Approximate x log data density."""
    with tf.variable_scope('xy_discriminator'):
        h = slim.repeat(x, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d, squeeze_dims=[1])


""" Construct model and training ops """
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=(batch_size, input_dim))
z = tf.placeholder(tf.float32, shape=(batch_size, latent_dim))
y = tf.placeholder(tf.float32, shape=(batch_size, input_dim))

# decoder and encoder
p_x = generative_network(z, input_dim , n_layer_gen, n_hidden_gen, eps_dim)
q_z = inference_network(x, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)

decoder_logit = data_network(p_x, z, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
encoder_logit = graph_replace(decoder_logit, {p_x: x, z:q_z})

decoder_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(decoder_logit), logits=decoder_logit)
encoder_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(encoder_logit), logits=encoder_logit)

disc_loss = tf.reduce_mean(  encoder_loss ) + tf.reduce_mean( decoder_loss)

rec_z = inference_network(p_x, latent_dim, n_layer_inf, n_hidden_inf, eps_dim )
rec_x = generative_network(q_z, input_dim , n_layer_gen, n_hidden_gen,  eps_dim )

cost_z = tf.reduce_mean(tf.pow(rec_z - z, 2))
cost_x = tf.reduce_mean(tf.pow(rec_x - x, 2))

decoder_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(decoder_logit), logits=decoder_logit)
encoder_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(encoder_logit), logits=encoder_logit)

gen_loss_xz = tf.reduce_mean(decoder_loss2) + tf.reduce_mean(encoder_loss2)

xy_distance = -tf.reduce_mean(tf.pow(tf.reduce_mean(y, reduction_indices=0) - tf.reduce_mean(p_x, reduction_indices=0),2))
gen_loss = 1.*gen_loss_xz + 1.0*cost_x + 1.0*cost_z

qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference")
pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative")
dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
xy_dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "xy_discriminator")

opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
train_gen_op = opt.minimize(gen_loss, var_list=qvars + pvars)
train_disc_op = opt.minimize(disc_loss, var_list=dvars)
train_xy_gen_op = opt.minimize(xy_distance, var_list=pvars)

""" training """
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


FG = []
FD = []
xy_FD = []

for epoch in tqdm( range(n_epoch), total=n_epoch):
    X_dataset= shuffle(X_dataset)
    Z_dataset= shuffle(Z_dataset)
    Y_dataset = shuffle(Y_dataset)
    i = 0
    for xmb, zmb, ymb in iter_data(X_dataset, Z_dataset, Y_dataset, size=batch_size):
        i = i + 1
        for _ in range(1):
            f_d, _ = sess.run([disc_loss, train_disc_op], feed_dict={x: xmb, z:zmb})
        for _ in range(5):
            f_g, _ = sess.run([[gen_loss, gen_loss_xz, cost_x, cost_z], train_gen_op], feed_dict={x: xmb, z: zmb, y: ymb})
        f_d_xy = sess.run(xy_distance, feed_dict={z: zmb, y: ymb})
        if f_d_xy > -10:
            sess.run(train_xy_gen_op, feed_dict={z: zmb, y: ymb})
        FG.append(f_g)
        FD.append(f_d)
        xy_FD.append(f_d_xy)
    print("epoch %d iter %d: discloss %f genloss %f adv_x %f recons_x %f recons_z %f xy_loss %f" % (epoch, i, f_d, f_g[0], f_g[1], f_g[2], f_g[3], f_d_xy))

""" plot the results """
# create Z test dataset
means_z = map(lambda x:  np.array(x), [[0, 0]])
means_z = list(means_z)
std_z = 1.0
variances_z = [np.eye(2) * std_z for _ in means_z]
priors_z = [1.0/len(means_z) for _ in means_z]

datasetZ_test = sample_GMM(dataset_size_z_test, means_z, variances_z, priors_z, sources=('features', ))
save_path = result_dir + 'Z_gmm_data_test.pdf'

# plot z test
Z_data_test = datasetZ_test.data['samples']
Z_targets_test = datasetZ_test.data['label']

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(Z_data_test[:, 0], Z_data_test[:, 1], c='r',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$z_1$'); ax.set_ylabel('$z_2$')
ax.axis('on')
plt.savefig(save_path, transparent=True, bbox_inches='tight')

# generate x
x_test = np.array([])
for zmb in iter_data(Z_data_test, size=batch_size):
    temp_x = sess.run(p_x, feed_dict={z: zmb})
    x_test = np.vstack([x_test, temp_x]) if x_test.size else temp_x

# plot x_test
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(x_test[:, 0], x_test[:, 1],c='b',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.axis('on')
save_path_x = result_dir + 'minusGAN_result.pdf'
plt.savefig(save_path_x, transparent=True, bbox_inches='tight')

# learning curves
fig_curve, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.plot(FD, label="Discriminator")
ax.plot(np.array(FG)[:,0], label="Generator")
ax.plot(np.array(FG)[:,1], label="Reconstruction x")
ax.plot(np.array(FG)[:,2], label="Reconstruction Z")
ax.plot(xy_FD, label="xy_distance")
plt.xlabel('Iteration')
plt.xlabel('Loss')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.axis('on')
plt.savefig(result_dir + 'learning_curves.pdf', bbox_inches='tight')


