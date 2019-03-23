import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from my_dl_lib import generic_dataset


class VanilaGan:
    """
    The most simplified gan, we will compare its results to the dcgan and more complicated ones.
    """
    def __init__(self, x_dim=784, model_name='vanila', loss_type='vanila'):

        self.tensor_board_logs_dir = os.path.join('logs', model_name)
        self.checkpoint_dir = os.path.join('saved_models', model_name)
        self.model_name = model_name
        self.x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name="x_input")

        # Discriminator:
        self.D_W1 = tf.Variable(self.xavier_init([x_dim, 128]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[128]))

        self.D_W2 = tf.Variable(self.xavier_init([128, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

        # Generator:
        self.generator_input_placeholder = tf.placeholder(tf.float32, shape=[None, 100])

        self.G_W1 = tf.Variable(self.xavier_init([100, 128]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[128]))

        self.G_W2 = tf.Variable(self.xavier_init([128, x_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[x_dim]))

        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        self.G_sample = self.init_generator()
        self.D_real, self.D_logit_real = self.init_discriminator(self.x_input)
        self.D_fake, self.D_logit_fake = self.init_discriminator(self.G_sample)
        if loss_type == 'vanila':
            self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1. - self.D_fake))
            self.G_loss = -tf.reduce_mean(tf.log(self.D_fake))

        # Alternative losses:
        # -------------------
        else:

            self.D_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_real, labels=tf.ones_like(self.D_logit_real)))
            self.D_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake, labels=tf.zeros_like(self.D_logit_fake)))
            self.D_loss = self.D_loss_real + self.D_loss_fake
            self.G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake, labels=tf.ones_like(self.D_logit_fake)))

        self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.theta_D)
        self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.theta_G)

        self.discriminator_loss_summary = tf.summary.scalar("Discriminator_loss", self.D_loss)
        self.generator_loss_summary = tf.summary.scalar("generator_loss", self.G_loss)

    @staticmethod
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    @staticmethod
    def plot(samples, type='mnist'):
        if type == 'mnist':
            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                if type == 'mnist':
                    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
                else:
                    plt.plot(sample)
            return fig
        else:
            fig = plt.figure()
            plt.plot(samples[0])
            # print(samples[0])
            return fig

    def init_generator(self):
        G_h1 = tf.nn.relu(tf.matmul(self.generator_input_placeholder, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob

    def init_discriminator(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1 ) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    def train(self, x_data=None, y_tags=None, number_of_iterations=1000000, batch_size=128):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Create a saver object :
        self.saver = tf.train.Saver()

        # Merge all the summaries and write them out:
        # self.merged_summaries = tf.summary.merge_all()
        self.generator_merged_summaries = tf.summary.merge([self.generator_loss_summary])

        self.discriminator_merged_summaries = tf.summary.merge([self.discriminator_loss_summary])

        # Create Writer object to visualize the graph later:
        self.generator_train_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/generator_train',
                                                            sess.graph)
        self.discriminator_train_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/discriminator_train',
                                                                sess.graph)

        # self.val_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/val')
        # self.test_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/test')

        i = 0

        if x_data is None:
            mnist = input_data.read_data_sets("mnist_data", one_hot=True)

        else:
            iterator_func = generic_dataset.GenericDataSetIterator(x_data, y_tags)

        for it in range(number_of_iterations):

            # x_batch = iterator_func.get_next_batch(x_data)
            if x_data is None:
                x_batch, _ = mnist.train.next_batch(128)
            else:
                x_batch, _ = iterator_func.next_batch(batch_size)
            generator_batch = np.random.uniform(-1., 1., size=[batch_size, 100])
            s, _, D_loss_curr = sess.run([self.discriminator_merged_summaries, self.D_solver, self.D_loss], feed_dict={self.x_input: x_batch,
                                                                               self.generator_input_placeholder:
                                                                                   generator_batch})
            self.discriminator_train_writer.add_summary(s, it)

            generator_batch = np.random.uniform(-1., 1., size=[batch_size, 100])
            s, _, G_loss_curr = sess.run([self.generator_merged_summaries, self.G_solver, self.G_loss], feed_dict={self.generator_input_placeholder:
                                                                                   generator_batch})

            self.generator_train_writer.add_summary(s, it)

            if it % 900 == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()

                samples = sess.run(self.G_sample, feed_dict={self.generator_input_placeholder: np.random.uniform
                (-1., 1., size=[16, 100])})

                if x_data is None:
                    fig = self.plot(samples, type='mnist')
                else:
                    fig = self.plot(samples, type='ecg')
                plt.savefig(os.path.join('samples', 'vanila_gan', self.model_name, '{}.png'.format(str(i).zfill(3))),
                            bbox_inches='tight')
                i += 1
                plt.close(fig)

                # Save the model :
                save_path = self.saver.save(sess, os.path.join(self.checkpoint_dir, 'iter_' + str(i)))
                print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    mnist_gan = VanilaGan(x_dim=784, model_name='mnist_2', loss_type='alternative')
    # mnist_data = input_data.read_data_sets("mnist_data", one_hot=True)
    # train_set = mnist_data.train
    mnist_gan.train()