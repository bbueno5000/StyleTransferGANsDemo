"""
DOCSTRING
"""
import data_gmm
import matplotlib.pyplot as pyplot
import numpy
import os
import random
import sklearn
import tensorflow
import tf_slim
import tqdm

tensorflow.compat.v1.disable_eager_execution()

class DataUtils:
    """
    DOCSTRING
    """
    def iter_data(self, *data, **kwargs):
        """
        DOCSTRING
        """
        size = kwargs.get('size', 128)
        try:
            n = len(data[0])
        except:
            n = data[0].shape[0]
        batches = int(n / size)
        if n % size != 0:
            batches += 1
        for b in range(batches):
            start = b * size
            end = (b + 1) * size
            if end > n:
                end = n
            if len(data) == 1:
                yield data[0][start:end]
            else:
                yield tuple([d[start:end] for d in data])

    def list_shuffle(self, *data):
        """
        DOCSTRING
        """
        idxs = RNG().np_rng.permutation(numpy.arange(len(data[0])))
        if len(data) == 1:
            return [data[0][idx] for idx in idxs]
        else:
            return [[d[idx] for idx in idxs] for d in data]

    def shuffle(self, *arrays, **options):
        """
        DOCSTRING
        """
        if isinstance(arrays[0][0], str):
            return self.list_shuffle(*arrays)
        else:
            return sklearn.utils.shuffle(*arrays, random_state=RNG().np_rng)

class RNG:
    """
    Random Number Generator
    """
    def __init__(self):
        self.seed = 42
        self.np_rng = numpy.random.RandomState(self.seed)
        self.py_rng = random.Random(self.seed)

    def set_seed(self, n):
        self.seed = n
        self.np_rng = numpy.random.RandomState(self.seed)
        self.py_rng = random.Random(self.seed)

class StyleTransferGAN:
    """
    DOCSTRING
    """
    def __init__(self):
        self.n_epoch = 1000
        self.batch_size  = 64
        dataset_size = 512
        self.input_dim = 2
        self.latent_dim = 2 
        self.eps_dim = 2
        # discriminator
        self.n_layer_disc = 2
        self.n_hidden_disc = 256
        # first generator 
        self.n_layer_gen = 2
        self.n_hidden_gen= 256
        # inference network/second generator
        self.n_layer_inf = 2
        self.n_hidden_inf= 256
        # save file
        self.result_dir = 'results/'
        directory = self.result_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        # first dataset
        means = map(lambda x:  numpy.array(x), [[0, 0], [2, 2], [-1, -1], [1, -1], [-1, 1]])
        means = list(means)
        std = 0.1
        variances = [numpy.eye(2) * std for _ in means]
        priors = [1.0/len(means) for _ in means]
        gaussian_mixture = data_gmm.Distribution(means=means, variances=variances, priors=priors)
        dataset = data_gmm.SampleGMM(dataset_size, means, variances, priors, sources=('features', ))
        save_path = self.result_dir + 'X_gmm_data.pdf'
        data_gmm.PlotGMM().plot(dataset, save_path)
        self.X_np_data = dataset.data['samples']
        self.X_labels = dataset.data['label']
        # second dataset
        means = map(lambda x:  numpy.array(x), [[-1, -1],[1, 1]])
        means = list(means)
        std = 0.1
        variances = [numpy.eye(2) * std for _ in means]
        priors = [1.0/len(means) for _ in means]
        gaussian_mixture = data_gmm.Distribution(means=means, variances=variances, priors=priors)
        dataset = data_gmm.SampleGMM(dataset_size, means, variances, priors, sources=('features', ))
        save_path = self.result_dir + 'Z_gmm_data.pdf'
        data_gmm.PlotGMM().plot(dataset, save_path)
        self.Z_np_data = dataset.data['samples']
        self.Z_labels = dataset.data['label']
        X_dataset = self.X_np_data
        Z_dataset = self.Z_np_data

    def __call__(self):
        tensorflow.compat.v1.reset_default_graph()
        # data1 input
        x = tensorflow.compat.v1.placeholder(
            tensorflow.float32, shape=(self.batch_size, self.input_dim))
        # data 2 input
        z = tensorflow.compat.v1.placeholder(
            tensorflow.float32, shape=(self.batch_size, self.latent_dim))
        # 2 generators - encoders
        p_x = self.generative_network(
            z, self.input_dim , self.n_layer_gen, self.n_hidden_gen, self.eps_dim)
        q_z = self.inference_network(
            x, self.latent_dim, self.n_layer_inf, self.n_hidden_inf, self.eps_dim)
        # The logit function is the inverse of the sigmoidal "logistic" function
        # 2 discriminators
        decoder_logit_x = self.data_network_x(
            p_x, n_layers=self.n_layer_disc, n_hidden=self.n_hidden_disc)
        encoder_logit_x = tensorflow.graph_editor.graph_replace(
            decoder_logit_x, {p_x: x})
        decoder_logit_z = self.data_network_z(
            q_z, n_layers=self.n_layer_disc, n_hidden=self.n_hidden_disc)
        encoder_logit_z = tensorflow.graph_editor.graph_replace(
            decoder_logit_z, {q_z: z})
        # Computes softplus: log(exp(features) + 1) activation for calculating loss
        encoder_sigmoid_x = tensorflow.nn.softplus(encoder_logit_x)
        decoder_sigmoid_x = tensorflow.nn.softplus(decoder_logit_x)
        encoder_sigmoid_z = tensorflow.nn.softplus(encoder_logit_z)
        decoder_sigmoid_z = tensorflow.nn.softplus(decoder_logit_z)
        # loss for both discriminators
        decoder_loss = decoder_sigmoid_x + decoder_sigmoid_z
        encoder_loss = encoder_sigmoid_x + encoder_sigmoid_z
        # combined loss for discriminators
        disc_loss = tensorflow.reduce_mean(encoder_loss) - tensorflow.reduce_mean(decoder_loss)
        # 2 more generators (decoders)
        rec_z = self.inference_network(
            p_x, self.latent_dim, self.n_layer_inf, self.n_hidden_inf, self.eps_dim)
        rec_x = self.generative_network(
            q_z, self.input_dim , self.n_layer_gen, self.n_hidden_gen, self.eps_dim)
        # compute generator loss
        # Sum of Squared Error loss
        cost_z = tensorflow.reduce_mean(tensorflow.pow(rec_z - z, 2))
        cost_x = tensorflow.reduce_mean(tensorflow.pow(rec_x - x, 2))
        # we tie in discriminator loss into generators loss
        adv_loss = tensorflow.reduce_mean(decoder_loss) 
        gen_loss = 1*adv_loss + 1.*cost_x  + 1.*cost_z
        # collect vars with names that contain this
        qvars = tensorflow.get_collection(
            tensorflow.GraphKeys.TRAINABLE_VARIABLES, "inference")
        pvars = tensorflow.get_collection(
            tensorflow.GraphKeys.TRAINABLE_VARIABLES, "generative")
        dvars_x = tensorflow.get_collection(
            tensorflow.GraphKeys.TRAINABLE_VARIABLES, "discriminator_x")
        dvars_z = tensorflow.get_collection(
            tensorflow.GraphKeys.TRAINABLE_VARIABLES, "discriminator_z")
        # use adam (gradient descent) to optimize
        opt = tensorflow.train.AdamOptimizer(1e-4, beta1=0.5)
        # minimize generators loss
        train_gen_op = opt.minimize(gen_loss, var_list=qvars + pvars)
        # minimize discirimaintors loss
        train_disc_op = opt.minimize(disc_loss, var_list=dvars_x+dvars_z)
        sess = tensorflow.InteractiveSession()
        sess.run(tensorflow.global_variables_initializer())
        FG, FD = [], []
        # for each epoch (log the status bar)
        for epoch in tqdm.tqdm(range(self.n_epoch), total=self.n_epoch):
            # sample from both our datasets
            X_dataset, Z_dataset = DataUtils.shuffle(X_dataset, Z_dataset)
            # for each x and z in our data 
            for xmb, zmb in DataUtils.iter_data(X_dataset, Z_dataset, size=self.batch_size):
                # minimize our loss functions
                for _ in range(1):
                    f_d, _ = sess.run([disc_loss, train_disc_op], feed_dict={x: xmb, z:zmb})
                for _ in range(5):
                    # 3 components that make up generator loss
                    f_g, _ = sess.run(
                        [[adv_loss, cost_x, cost_z], train_gen_op], feed_dict={x: xmb, z:zmb})
                FG.append(f_g)
                FD.append(f_d)
        n_viz = 1
        imz = numpy.array([]); rmz = numpy.array([]); imx = numpy.array([]); rmx = numpy.array([]);
        for _ in range(n_viz):
            for xmb, zmb in DataUtils.iter_data(self.X_np_data, self.Z_np_data, size=self.batch_size):
                temp_imz = sess.run(q_z, feed_dict={x: xmb, z:zmb})
                imz = numpy.vstack([imz, temp_imz]) if imz.size else temp_imz
                temp_rmz = sess.run(rec_z, feed_dict={x: xmb, z:zmb})
                rmz = numpy.vstack([rmz, temp_rmz]) if rmz.size else temp_rmz
                temp_imx = sess.run(p_x, feed_dict={x: xmb, z:zmb})
                imx = numpy.vstack([imx, temp_imx]) if imx.size else temp_imx
                temp_rmx = sess.run(rec_x, feed_dict={x: xmb, z:zmb})
                rmx = numpy.vstack([rmx, temp_rmx]) if rmx.size else temp_rmx
        # inferred marginal z
        fig_mz, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
        ll = numpy.tile(self.X_labels, (n_viz))
        ax.scatter(
            imz[:, 0], imz[:, 1], c=pyplot.cm.Set1(ll.astype(float)/self.input_dim/2.0),
            edgecolor='none', alpha=0.5)
        ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel('$z_1$'); ax.set_ylabel('$z_2$')
        ax.axis('on')
        pyplot.savefig(
            self.result_dir + 'inferred_mz.pdf', transparent=True, bbox_inches='tight')
        #  reconstructed z
        fig_pz, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
        ll = numpy.tile(self.Z_labels, (n_viz))
        ax.scatter(
            rmz[:, 0], rmz[:, 1],
            c=pyplot.cm.Set1(ll.astype(float)/self.input_dim/2.0),
            edgecolor='none', alpha=0.5)
        ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel('$z_1$'); ax.set_ylabel('$z_2$')
        ax.axis('on')
        pyplot.savefig(
            self.result_dir + 'reconstruct_mz.pdf', transparent=True, bbox_inches='tight')
        # inferred marginal x
        fig_pz, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
        ll = numpy.tile(self.Z_labels, (n_viz))
        ax.scatter(
            imx[:, 0], imx[:, 1],
            c=pyplot.cm.Set1(ll.astype(float)/self.input_dim/2.0),
            edgecolor='none', alpha=0.5)
        ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.axis('on')
        pyplot.savefig(
            self.result_dir + 'inferred_mx.pdf', transparent=True, bbox_inches='tight')
        #  reconstruced x
        fig_mx, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
        ll = numpy.tile(self.X_labels, (n_viz))
        ax.scatter(
            rmx[:, 0], rmx[:, 1],
            c=pyplot.cm.Set1(ll.astype(float)/self.input_dim/2.0),
            edgecolor='none', alpha=0.5)
        ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.axis('on')
        pyplot.savefig(
            self.result_dir + 'reconstruct_mx.pdf', transparent=True, bbox_inches='tight')
        # learning curves
        fig_curve, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
        ax.plot(FD, label="Discriminator")
        ax.plot(numpy.array(FG)[:,0], label="Generator")
        ax.plot(numpy.array(FG)[:,1], label="Reconstruction x")
        ax.plot(numpy.array(FG)[:,2], label="Reconstruction Z")
        pyplot.xlabel('Iteration')
        pyplot.xlabel('Loss')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.axis('on')
        pyplot.savefig(
            self.result_dir + 'learning_curves.pdf', bbox_inches='tight')

    def data_network_x(self, x, n_layers=2, n_hidden=256, activation_fn=None):
        """
        Approximate x log data density.
        """
        h = tensorflow.concat(x, 1)
        with tensorflow.compat.v1.variable_scope('discriminator_x'):
            h = tf_slim.repeat(
                h, n_layers, tf_slim.fully_connected,
                n_hidden, activation_fn=tensorflow.nn.relu)
            log_d = tf_slim.fully_connected(h, 1, activation_fn=activation_fn)
        return tensorflow.squeeze(log_d, axis=[1])

    def data_network_z(self, z, n_layers=2, n_hidden=256, activation_fn=None):
        """
        Approximate z log data density.
        """
        h = tensorflow.concat(z, 1)
        with tensorflow.compat.v1.variable_scope('discriminator_z'):
            h = tf_slim.repeat(
                h, n_layers, tf_slim.fully_connected,
                n_hidden, activation_fn=tensorflow.nn.relu)
            log_d = tf_slim.fully_connected(h, 1, activation_fn=activation_fn)
        return tensorflow.squeeze(log_d, axis=[1])

    def generative_network(self, z, input_dim, n_layer, n_hidden, eps_dim):
        """
        DOCSTRING
        """
        with tensorflow.compat.v1.variable_scope("generative"):
            h = z
            h = tf_slim.repeat(
                h, n_layer, tf_slim.fully_connected,
                n_hidden, activation_fn=tensorflow.nn.relu)
            x = tf_slim.fully_connected(h, input_dim, activation_fn=None, scope="p_x")
        return x

    def inference_network(self, x, latent_dim, n_layer, n_hidden, eps_dim):
        """
        DOCSTRING
        """
        with tensorflow.compat.v1.variable_scope("inference"):
            h = x
            h = tf_slim.repeat(
                h, n_layer, tf_slim.fully_connected,
                n_hidden, activation_fn=tensorflow.nn.relu)
            z = tf_slim.fully_connected(h, latent_dim, activation_fn=None, scope="q_z")
        return z

if __name__ == '__main__':
    style_transfer_gan = StyleTransferGAN()
    style_transfer_gan()
