from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.nn import softplus
from edward.models import Bernoulli, Normal, Gamma, TransformedDistribution
import pandas as pd
import ghalton as gh
from sobol_seq import i4_sobol_generate as sobol_gen

tb_log_dir = None


def paired_squared_euclidean(A, B):
    A_norm = tf.norm(A, ord='euclidean', axis=1, keepdims=True)
    B_norm = tf.norm(B, ord='euclidean', axis=1, keepdims=True)
    return tf.square(A_norm) - \
           2 * tf.matmul(A, B, transpose_b=True) + \
           tf.square(tf.transpose(B_norm))


def rbf_kernel(X, Y, gamma, tfdt, batch_size=None):
    """
    rbf features (hinged rbf when Y are grid locations)
    :param X:     data
    :param Y:     kernel locations
    :param gamma: kernel parameter
    :param tfdt:  tensorflow datatype
    :return:
    """
    if batch_size is None:
        batch_size = X.shape[0].value
    dist = tf.exp(-gamma * paired_squared_euclidean(X, Y))
    features = tf.concat(
        [tf.ones(shape=(batch_size, 1), dtype=tfdt), dist], axis=1)
    return features


def calc_grid_v2(cell_resolution, max_min, method='grid', X=None, M=None):
    """
    :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
    :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max)
    :param X: a sample of lidar locations
    :return: numpy array of size (# of RNFs, 2) with grid locations
    """
    if max_min is None:
        # if 'max_min' is not given, make a boundarary based on X
        # assume 'X' contains samples from the entire area
        expansion_coef = 1.2
        x_min, x_max = expansion_coef * X[:, 0].min(), expansion_coef * X[:,
                                                                        0].max()
        y_min, y_max = expansion_coef * X[:, 1].min(), expansion_coef * X[:,
                                                                        1].max()
    else:
        x_min, x_max = max_min[0], max_min[1]
        y_min, y_max = max_min[2], max_min[3]

    if method == 'grid':  # on a regular grid
        xvals = np.arange(x_min, x_max, cell_resolution[0])
        yvals = np.arange(y_min, y_max, cell_resolution[1])
        xx, yy = np.meshgrid(xvals, yvals)
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
    else:  # sampling
        D = 2
        if M is None:
            xsize = np.int((x_max - x_min) / cell_resolution[0])
            ysize = np.int((y_max - y_min) / cell_resolution[1])
            M = np.int((x_max - x_min) / cell_resolution[0]) * np.int(
                (y_max - y_min) / cell_resolution[1])
        if method == 'mc':
            grid = np.random.uniform(0, 1, (M, D))
        elif method == 'halton':
            grid = np.array(gh.Halton(D).get(M))
        elif method == 'ghalton':
            grid = np.array(gh.GeneralizedHalton(gh.EA_PERMS[:D]).get(int(M)))
        elif method == 'sobol':
            grid = sobol_gen(D, M, 7)
        else:
            grid = None

        grid[:, 0] = x_min + (x_max - x_min) * grid[:, 0]
        grid[:, 1] = y_min + (y_max - y_min) * grid[:, 1]

    return grid


def load_parameters(case):
    parameters = \
        {
         'dataset1': \
             (
                 'datasets/dataset1.csv',
                 (4, 7),
                 (150, 200, 0, 300),
                 1,
                 0.0,
                 0.0
             ),
         }

    return parameters[case]


def lognormal_q(shape, name=None):
    with tf.variable_scope(name, default_name="lognormal_q"):
        min_scale = 1e-5
        loc = tf.get_variable("loc", shape,
                              initializer=tf.random_normal_initializer(
                                  mean=1.0,
                                  stddev=0.1, ),
                              trainable=True)
        scale = tf.get_variable(
            "scale", shape,
            initializer=tf.random_normal_initializer(
                mean=-3.0,
                stddev=0.1),
            trainable=True)
        rv = TransformedDistribution(
            distribution=Normal(loc, tf.maximum(softplus(scale), min_scale)),
            bijector=tf.contrib.distributions.bijectors.Exp())
        return rv


if __name__ == "__main__":
    tfdt = tf.float32
    fn_train, cell_res, cell_minmax, skip, thresh, _ = load_parameters(
        'dataset1')

    # read data and split into to training test
    g = pd.read_csv(fn_train, delimiter=',').values
    filter_locs = list(set(np.where(g[:, 1] > 150)[0]).intersection(
        set(np.where(g[:, 1] < 200)[0])))  # Filter area
    g = g[filter_locs, :]
    X_all = np.float_(g[:, 0:3])
    Y_all = np.float_(g[:, 3][:, np.newaxis]).ravel()
    X_hinge_grid = calc_grid_v2(cell_res, cell_minmax, method='ghalton',
                                X=X_all)

    N, D = X_all.shape[0], 2  # -1 because ignore time dim 1
    M = X_hinge_grid.shape[0] + 1
    print("Data dimensionality is: {}".format(D))
    print("Hinged feature dimensionality is: {}".format(M))
    # DATA
    X = value = X_all[:, 1:]
    Y = Y_all.reshape(-1, 1)
    hinge_grid = Normal(loc=X_hinge_grid * tf.ones(shape=X_hinge_grid.shape,
                                                   dtype=tfdt),
                        scale=10 * tf.ones(shape=X_hinge_grid.shape,
                                           dtype=tfdt))
    qhinge_grid = Normal(
        loc=tf.get_variable("qhinge_loc", shape=X_hinge_grid.shape,
                            initializer=tf.random_normal_initializer(
                                mean=X_hinge_grid * tf.ones(
                                    shape=X_hinge_grid.shape,
                                    dtype=tfdt),
                                stddev=0.001 * tf.ones(shape=X_hinge_grid.shape,
                                                       dtype=tfdt))),
        scale=softplus(tf.get_variable("qhinge_scale", shape=X_hinge_grid.shape,
                                       initializer=tf.random_normal_initializer(
                                           mean=2.0,
                                           stddev=0.1))))

    gamma_shape = 1.05
    gamma_rate = 12.0

    gamma = Gamma(gamma_shape, gamma_rate, sample_shape=[1, M - 1])
    qgamma = lognormal_q([1, M - 1])

    # Model definition
    X_plh = tf.placeholder(tfdt, shape=[N, D])
    PHI = rbf_kernel(X=X_plh, Y=hinge_grid, gamma=gamma, tfdt=tfdt)
    w = Normal(loc=tf.zeros((M, 1), dtype=tfdt),
               scale=900 * tf.ones((M, 1), dtype=tfdt))
    y = Bernoulli(logits=tf.matmul(PHI, w))

    # Variational distributions
    qw_loc = tf.get_variable("qw_loc", [M, 1], dtype=tfdt)
    qw_scale = softplus(120 * tf.get_variable("qw_scale", [M, 1], dtype=tfdt))
    qw = Normal(loc=qw_loc, scale=qw_scale)

    # Inference settings
    n_samples = 19
    n_iters = 2450
    learning_rate = 1e-2
    beta1 = 0.9
    beta2 = 0.999
    adam_epsilon = 1e-8
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=adam_epsilon)
    inference = ed.KLqp({w: qw,
                         gamma: qgamma,
                         hinge_grid: qhinge_grid},
                        data={X_plh: X, y: Y})
    inference.initialize(n_samples=n_samples,
                         n_iter=n_iters,
                         n_print=10,
                         optimizer=optimizer,
                         logdir=tb_log_dir)

    tf.global_variables_initializer().run()

    display_xlim = (145, 205)
    display_ylim = (-10, 310)
    title_fontsize = 20
    colorbar_labelsize = 15
    axes_labelsize = 15
    suptitle_fontsize = 25

    contourf_res = 100
    qcellres = (2, 2)
    qminmax = (150, 200, 0, 300)
    qmesh_x1, qmesh_x2 = np.meshgrid(
        np.linspace(qminmax[0], qminmax[1] - qcellres[0],
                    (qminmax[1] - qminmax[0]) // qcellres[0]),
        np.linspace(qminmax[2], qminmax[3] - qcellres[1],
                    (qminmax[3] - qminmax[2]) // qcellres[1]))

    gamma_mesh_x1, gamma_mesh_x2 = np.meshgrid(
        np.linspace(cell_minmax[0], cell_minmax[1] - cell_res[0],
                    (cell_minmax[1] - cell_minmax[0]) // cell_res[0]),
        np.linspace(cell_minmax[2], cell_minmax[3] - cell_res[1],
                    (cell_minmax[3] - cell_minmax[2]) // cell_res[1]))

    X_q = calc_grid_v2(qcellres, qminmax, method='grid', X=None)
    X_q_tf = tf.constant(X_q, dtype=tfdt)
    X_q_features = rbf_kernel(X_q_tf, qhinge_grid.mean(),
                              qgamma.bijector.forward(
                                  qgamma.distribution.mean()), tfdt)

    # Running inference
    for t in range(inference.n_iter):
        if t % 10 == 0:
            print("\nsaving {}".format(t))
            qgamma_eval = qgamma.bijector.forward(
                qgamma.distribution.mean()).eval()
            qgamma_var_eval = qgamma.distribution.variance().eval()
            qhinge_grid_eval = qhinge_grid.mean().eval()
            post_mu = tf.matmul(X_q_features, qw.mean())
            post_var = tf.reduce_sum(
                tf.square(X_q_features) * tf.transpose(qw.variance()),
                axis=1, keepdims=True)
            kappa_var = 1 / tf.sqrt(1 + np.pi * post_var / 8)
            probs = tf.sigmoid(kappa_var * post_mu)
            probs_eval = probs.eval()

            FIG = plt.figure(figsize=(14, 9))

            ax = plt.subplot(131)
            scattr = ax.scatter(X[:, 0], X[:, 1], c=Y.flatten(), s=2,
                                cmap='jet')
            ax.set_xlim(*display_xlim)
            ax.set_ylim(*display_ylim)
            cbar = plt.colorbar(scattr, ticks=[0, 1])
            cbar.set_ticklabels(['0', '1'])
            cbar.ax.tick_params(labelsize=colorbar_labelsize)
            ax.tick_params(axis='both', labelsize=axes_labelsize)
            ax.set_title('Ground truth', fontsize=title_fontsize)

            ax = plt.subplot(132)
            contf = ax.contourf(qmesh_x1, qmesh_x2,
                                probs_eval.reshape(qmesh_x1.shape),
                                contourf_res, cmap='jet', vmin=0, vmax=1)
            ax.set_xlim(*display_xlim)
            ax.set_ylim(*display_ylim)
            occupancy_ticks = np.linspace(0.0, 1.0, 5)
            cbar = plt.colorbar(contf, ticks=occupancy_ticks)

            cbar.ax.tick_params(labelsize=colorbar_labelsize)
            ax.tick_params(axis='both', labelsize=axes_labelsize)
            ax.set_title('Predicted occupancy', fontsize=title_fontsize)

            ax = plt.subplot(133)
            sortidxs = qgamma_eval.argsort()
            qgamma_eval = qgamma_eval.T[sortidxs]
            qhinge_grid_eval[:, 0] = qhinge_grid_eval[:, 0][sortidxs].flatten()
            qhinge_grid_eval[:, 1] = qhinge_grid_eval[:, 1][sortidxs].flatten()
            smin = 5 * 2
            smax = 230 * 2
            lengthscale_cmap = "rainbow_r"
            logls = np.log(qgamma_eval)
            loglsmax = np.min(logls)
            loglsmin = np.max(logls)
            srange = smax - smin
            loglsrange = loglsmax - loglsmin
            logls = (logls - loglsmin) / loglsrange
            logls = (logls * srange) + smin
            scattr = ax.scatter(qhinge_grid_eval[:, 0], qhinge_grid_eval[:, 1],
                                c=qgamma_eval.flatten(),
                                s=logls, marker="o", lw=0.0,
                                cmap=lengthscale_cmap,
                                alpha=0.85)
            ax.set_xlim(*display_xlim)
            ax.set_ylim(*display_ylim)

            qgamma_ticks = np.linspace(qgamma_eval.min(), qgamma_eval.max(), 4)
            sbar = plt.colorbar(scattr, ticks=qgamma_ticks)
            tick_labels = ["{:.1f}".format(np.sqrt(1 / (2 * val))) for val in
                           qgamma_ticks]
            sbar.set_ticklabels(tick_labels)
            sbar.ax.invert_yaxis()
            sbar.ax.tick_params(labelsize=colorbar_labelsize)
            ax.tick_params(axis='both', labelsize=axes_labelsize)
            ax.set_title('Learned kernels\n(positions and lengthscales)',
                         fontsize=title_fontsize)

            plt.tight_layout()
            plt.suptitle("Iteration: {}\n".format(t),
                         fontsize=suptitle_fontsize)
            FIG.subplots_adjust(top=0.85)
            plt.savefig('figs_main_anim/{:000003d}.png'.format(t), dpi=200,
                        format='png')

        info_dict = inference.update()
        inference.print_progress(info_dict)
