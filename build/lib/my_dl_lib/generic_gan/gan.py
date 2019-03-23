import os
import re
from glob import glob

import my_dl_lib.generic_dataset
import matplotlib.gridspec as gridspec
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.misc
import tensorflow as tf
from my_dl_lib.generic_gan import discriminator
from my_dl_lib.generic_gan import generator
from my_dl_lib.generic_gan import input_data


class GenericGAN:

    def __init__(self, discriminator_input_height, discriminator_input_width, discriminator_number_of_input_channels=3,
                 discriminator_number_of_filters_at_first_layer=64, discriminator_arch_file=None,
                 generator_input_dim=100, generator_output_height=28, generator_output_width=28,
                 generator_output_channels=1, generator_architecture_file=None, tensor_board_logs_dir='./logs',
                 is_input_an_image=True, checkpoint_dir='./checkpoints/gan', generated_samples_dir='GAN_samples',
                 y_labels_dim=None, with_activation_function_on_last_layer_of_generator=False, gan_type='dcgan'):

        self.gan_type = gan_type
        self.num_of_iterations_traind = 0  # Helper variable to count how many training iterations we did so far.
        self.saver = None
        self.checkpoint_dir = checkpoint_dir
        self.generated_samples_dir = generated_samples_dir
        self.is_input_an_image = is_input_an_image
        self.tensor_board_logs_dir = tensor_board_logs_dir

        self.generator = generator.GenericGenerator(generator_input_dim, generator_output_height,
                                                    generator_output_width, generator_output_channels,
                                                    generator_architecture_file,
                                                    y_labels_dim=y_labels_dim,
                                                    add_activation_at_end=
                                                    with_activation_function_on_last_layer_of_generator,
                                                    gan_type=self.gan_type)

        self.y_place_holder = tf.placeholder(tf.float32, shape=[None, y_labels_dim], name='y_input_placeholder')

        self.discriminator = discriminator.GenericDiscriminator(discriminator_input_height, discriminator_input_width,
                                                                discriminator_number_of_input_channels,
                                                                discriminator_number_of_filters_at_first_layer,
                                                                discriminator_arch_file,
                                                                y_labels_dim=y_labels_dim,
                                                                y_placeholder=self.y_place_holder,
                                                                gan_type=self.gan_type)

        generator_output = self.generator.last_layer_generator_tensor
        self.discriminator_of_generator = discriminator.GenericDiscriminator(discriminator_input_height, discriminator_input_width,
                                                                discriminator_number_of_input_channels,
                                                                discriminator_number_of_filters_at_first_layer,
                                                                discriminator_arch_file, inputs=generator_output,
                                                                             y_labels_dim=y_labels_dim,
                                                                             y_placeholder=self.y_place_holder,
                                                                             gan_type=self.gan_type)

        self.total_discriminator_loss = self.discriminator.cost + self.discriminator_of_generator.cost
        self.total_discriminator_loss_summary = tf.summary.scalar("Discriminator_total_loss",
                                                                  self.total_discriminator_loss)

        '''
        Once we have our 2 loss functions (d_loss and g_loss), we need to define our optimizers. Keep in mind that the 
        optimizer for the generator network needs to only update the generator’s weights, not those of the 
        discriminator. In order to make this distinction, we need to create 2 lists, one with the discriminator’s 
        weights and one with the generator’s weights. This is where naming all of your Tensorflow variables can come 
        in handy.
        '''
        t_vars = tf.trainable_variables()
        # TODO: Delete this.
        print("Trainable Vars:")
        for i, var in enumerate(t_vars):
            print("%d : " % i, end='')
            print(var)

        self.discriminator_vars = [var for var in t_vars if 'discriminator' in var.name]
        print("Discriminator Vars:")
        for i, var in enumerate(self.discriminator_vars):
            print("%d : " % i, end='')
            print(var)
        self.generator_vars = [var for var in t_vars if 'generator' in var.name]
        print("Generator Vars:")
        for i, var in enumerate(self.generator_vars):
            print("%d : " % i, end='')
            print(var)
        print("Done")

    def train(self, discriminator_x_train, discriminator_y_train,
              num_of_iterations, batch_size, learning_rate=0.0002,
              momentum_term_adam=0.5, crop_images=False):
        """

        :return:
        """
        with tf.variable_scope("train"):
            discriminator_updates = tf.train.AdamOptimizer(learning_rate, beta1=momentum_term_adam) \
                .minimize(self.total_discriminator_loss, var_list=self.discriminator_vars)

            generator_updates = tf.train.AdamOptimizer(learning_rate, beta1=momentum_term_adam) \
                .minimize(self.discriminator_of_generator.generator_cost, var_list=self.generator_vars)

        self.session = tf.Session()
        sess = self.session
        init = tf.global_variables_initializer()
        sess.run(init)

        # Create a saver object :
        self.saver = tf.train.Saver(max_to_keep=1000)

        # Merge all the summaries and write them out:
        # self.merged_summaries = tf.summary.merge_all()
        self.generator_merged_summaries = tf.summary.merge([self.discriminator_of_generator.loss_summary,
                                                            self.discriminator_of_generator.generator_loss_summary])

        self.discriminator_merged_summaries = tf.summary.merge([self.discriminator.loss_summary,
                                                               self.total_discriminator_loss_summary])

        # Create Writer object to visualize the graph later:
        self.generator_train_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/generator_train',
                                                            sess.graph)
        self.discriminator_train_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/discriminator_train',
                                                                sess.graph)

        self.val_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/val')
        self.test_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/test')

        # generator_iterator = generic_dataset.GenericDataSetIterator(generator_x_train, generator_y_train)
        discriminator_iterator = my_dl_lib.generic_dataset.GenericDataSetIterator(discriminator_x_train, discriminator_y_train)
        j = 0
        for i in range(num_of_iterations):
            # generator_x_batch, generator_y_batch = generator_iterator.next_batch(batch_size)
            generator_x_batch = np.random.uniform(-1, 1, [batch_size, self.generator.input_dim]).astype(np.float32)

            discriminator_x_batch, discriminator_y_batch = discriminator_iterator.next_batch(batch_size)
            if crop_images:
                discriminator_x_batch = np.array([get_image(img_file,
                              input_height=108,
                              input_width=108,
                              resize_height=64,
                              resize_width=64,
                              crop=True,
                              grayscale=False) for img_file in discriminator_x_batch])

            '''
            while j < 500:  # Start by only updating the discriminator:
                # Update the Discriminator network :
                print("**Iterating on discriminator Only. Iteration %d**", j)
                summary, _ = sess.run([self.discriminator_merged_summaries, discriminator_updates],
                                      feed_dict={self.discriminator.input_placeholder: discriminator_x_batch,
                                                 self.generator.x_input_placeholder: generator_x_batch})
                j += 1
                self.discriminator_train_writer.add_summary(summary, i)
                generator_x_batch = np.random.uniform(-1, 1, [batch_size, self.generator.input_dim]).astype(np.float32)

                discriminator_x_batch, _ = discriminator_iterator.next_batch(batch_size)
            '''
            # Update the Discriminator network :
            if self.is_conditional_gan:
                summary, _ = sess.run([self.discriminator_merged_summaries, discriminator_updates],
                                      feed_dict={self.discriminator.input_placeholder: discriminator_x_batch,
                                                 self.generator.x_input_placeholder: generator_x_batch,
                                                 self.discriminator.y_input_placeholder: discriminator_y_batch,
                                                 self.generator.y_input_placeholder: discriminator_y_batch})

                self.discriminator_train_writer.add_summary(summary, i)

                # Update Generator network:
                summary, _ = sess.run([self.generator_merged_summaries, generator_updates],
                                      feed_dict={self.generator.x_input_placeholder: generator_x_batch,
                                                 self.generator.y_input_placeholder: discriminator_y_batch,
                                                 self.discriminator_of_generator.y_input_placeholder: discriminator_y_batch})

                self.generator_train_writer.add_summary(summary, i)

                # Update Generator twice to make sure that d_loss does not go to zero (different from paper)
                generator_loss, summary, _ = sess.run(
                    [self.discriminator_of_generator.generator_cost, self.generator_merged_summaries,
                     generator_updates],
                    feed_dict={self.generator.x_input_placeholder: generator_x_batch,
                               self.generator.y_input_placeholder: discriminator_y_batch,
                               self.discriminator_of_generator.y_input_placeholder: discriminator_y_batch})
            else:
                summary, _ = sess.run([self.discriminator_merged_summaries, discriminator_updates],
                                      feed_dict={self.discriminator.input_placeholder: discriminator_x_batch,
                                                 self.generator.x_input_placeholder: generator_x_batch})

                self.discriminator_train_writer.add_summary(summary, i)

                # Update Generator network:
                summary, _ = sess.run([self.generator_merged_summaries, generator_updates],
                                      feed_dict={ self.generator.x_input_placeholder: generator_x_batch})

                self.generator_train_writer.add_summary(summary, i)

                # Update Generator twice to make sure that d_loss does not go to zero (different from paper)
                generator_loss, summary, _ = sess.run([self.discriminator_of_generator.generator_cost, self.generator_merged_summaries, generator_updates],
                                      feed_dict={self.generator.x_input_placeholder: generator_x_batch})

            '''
            while generator_loss > 2:
                generator_x_batch = np.random.uniform(-1, 1, [batch_size, self.generator.input_dim]).astype(np.float32)
                generator_loss, summary, _ = sess.run(
                    [self.discriminator_of_generator.generator_cost, self.generator_merged_summaries,
                     generator_updates],
                    feed_dict={self.generator.x_input_placeholder: generator_x_batch})
                print("inside while, generator loss is ", generator_loss)
            '''
            self.generator_train_writer.add_summary(summary, i)

            if i % 10 == 0:

                if self.is_conditional_gan:
                    loss_discriminator_from_generator_input = sess.run(self.discriminator_of_generator.cost,
                                                                       feed_dict={self.generator.x_input_placeholder:
                                                                                      generator_x_batch,
                                                                                  self.generator.y_input_placeholder:
                                                                                      discriminator_y_batch,
                                                                                  self.discriminator_of_generator.y_input_placeholder:
                                                                                      discriminator_y_batch})

                    loss_discriminator_from_real_input = sess.run(self.discriminator.cost,
                                                                  feed_dict={self.discriminator.input_placeholder:
                                                                                 discriminator_x_batch,
                                                                             self.discriminator.y_input_placeholder: discriminator_y_batch})

                    loss_generator = sess.run(self.discriminator_of_generator.generator_cost,
                                              feed_dict={self.generator.x_input_placeholder: generator_x_batch,
                                                         self.generator.y_input_placeholder: discriminator_y_batch,
                                                         self.discriminator_of_generator.y_input_placeholder: discriminator_y_batch})
                else:
                    loss_discriminator_from_generator_input = sess.run(self.discriminator_of_generator.cost,
                                                                       feed_dict={self.generator.x_input_placeholder:
                                                                                      generator_x_batch})

                    loss_discriminator_from_real_input = sess.run(self.discriminator.cost,
                                                                  feed_dict={self.discriminator.input_placeholder:
                                                                                 discriminator_x_batch})

                    loss_generator = sess.run(self.discriminator_of_generator.generator_cost,
                                              feed_dict={self.generator.x_input_placeholder: generator_x_batch})

                print("Iteration number %d, discriminator_loss: %.8f, generator_loss: %.8f" %
                      (i, loss_discriminator_from_generator_input + loss_discriminator_from_real_input, loss_generator))

            if i % 100 == 0:
                # Save the model :
                save_path = self.saver.save(sess, os.path.join(self.checkpoint_dir, 'iter_' + str(i)))
                print("Model saved in file: %s" % save_path)

                if self.is_input_an_image:
                    # sample from the generator and plot :
                    random_samples = np.random.uniform(-1, 1, size=(64, self.generator.input_dim))
                    y_samples = discriminator_y_train[:64]
                    output_images = self.generator.run_generator(sess, random_samples, y_samples)
                    # self.plot(output_images, type='mnist')

                    manifold_h = int(np.floor(np.sqrt(64)))
                    manifold_w = int(np.ceil(np.sqrt(64)))
                    assert manifold_h * manifold_w == 64
                    size = manifold_h, manifold_w

                    self.save_batch_of_images(output_images, size, os.path.join(self.generated_samples_dir,
                                                                                'sample_iter_' + str(i) + '.png'))
                    '''
                    if output_images[0].shape[2] == 1:
                        fig = plt.figure()
                        plt.imshow(output_images[0][:, :, 0], cmap='Greys_r')
                        plt.savefig(os.path.join(self.generated_samples_dir, str(i) + "_mnist.png"))
                    else:
                        # RGB image:
                        assert output_images[0].shape[2] == 3
                        manifold_h = int(np.floor(np.sqrt(64)))
                        manifold_w = int(np.ceil(np.sqrt(64)))
                        assert manifold_h * manifold_w == 64
                        size = manifold_h, manifold_w
                        self.save_batch_of_images(output_images, size, os.path.join(self.generated_samples_dir,
                                                                                    'sample_iter_' + str(i) + '.png'))
                    '''
                else:
                    # sample from the generator and plot :
                    random_samples = np.random.uniform(-1, 1, size=(5, self.generator.input_dim))
                    # Valid only for ECG - Generate for each beat type 5 samples:
                    N_sample = [1, 0, 0, 0, 0]
                    S_sample = [0, 1, 0, 0, 0]
                    V_sample = [0, 0, 1, 0, 0]
                    F_sample = [0, 0, 0, 1, 0]
                    Q_sample = [0, 0, 0, 0, 1]

                    # y_samples = discriminator_y_train[:5]
                    y_samples = np.array([N_sample, S_sample, V_sample, F_sample, Q_sample])
                    output_generator = self.generator.run_generator(sess, random_samples, y_samples)
                    plt.figure()
                    plt.subplot(3, 2, 1)
                    plt.plot(output_generator[0])
                    plt.title("N sample from generator")
                    plt.subplot(3, 2, 2)
                    plt.plot(output_generator[1])
                    plt.title("S sample from generator")
                    plt.subplot(3, 2, 3)
                    plt.plot(output_generator[2])
                    plt.title("V sample from generator")
                    plt.subplot(3, 2, 4)
                    plt.plot(output_generator[3])
                    plt.title("F sample from generator")
                    plt.subplot(3, 2, 5)
                    plt.plot(output_generator[4])
                    plt.title("Q sample from generator")
                    plt.savefig(os.path.join(self.generated_samples_dir, str(i) + "_ecg_gan.png"))

                '''
                for j,e in enumerate(output_generator):
                    # print(e)
                    plt.figure()
                    plt.plot(e)
                    plt.savefig(os.path.join(self.generated_samples_dir, str(i) + str(j) + "_ecg_gan.png"))
                    # plt.show()
                '''

    def continue_training_from_meta(self, model_name, discriminator_x_train, discriminator_y_train, num_of_iterations,
                                    batch_size):

        # Load the parameters:
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(model_name + 'meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('./'))
            graph = tf.get_default_graph()

            with tf.variable_scope("train"):
                discriminator_updates = graph.get_tensor_by_name("discriminator_updates:0")

                generator_updates = graph.get_tensor_by_name("generator_updates:0")

            # Create a saver object :
            self.saver = tf.train.Saver()

            # Merge all the summaries and write them out:
            # self.merged_summaries = tf.summary.merge_all()
            self.generator_merged_summaries = graph.get_tensor_by_name("generator_merged_summaries:0")

            self.discriminator_merged_summaries = graph.get_tensor_by_name("discriminator_merged_summaries:0")

            # Create Writer object to visualize the graph later:
            self.generator_train_writer = graph.get_tensor_by_name("generator_train_writer:0")

            self.discriminator_train_writer = graph.get_tensor_by_name("discriminator_train_writer:0")

            discriminator_iterator = my_dl_lib.generic_dataset.GenericDataSetIterator(discriminator_x_train,
                                                                            discriminator_y_train)
            for i in range(num_of_iterations):
                # generator_x_batch, generator_y_batch = generator_iterator.next_batch(batch_size)
                generator_x_batch = np.random.uniform(-1, 1, [batch_size, self.generator.input_dim]).astype(np.float32)

                discriminator_x_batch, _ = discriminator_iterator.next_batch(batch_size)

                # Update the Discriminator network:
                summary, _ = sess.run([self.discriminator_merged_summaries, discriminator_updates],
                                      feed_dict={self.discriminator.input_placeholder: discriminator_x_batch,
                                                 self.generator.x_input_placeholder: generator_x_batch})

                self.discriminator_train_writer.add_summary(summary, i)

                # Update Generator network:
                summary, _ = sess.run([self.generator_merged_summaries, generator_updates],
                                      feed_dict={self.generator.x_input_placeholder: generator_x_batch})

                self.generator_train_writer.add_summary(summary, i)

                # Update Generator twice to make sure that d_loss does not go to zero (different from paper)
                summary, _ = sess.run([self.generator_merged_summaries, generator_updates],
                                      feed_dict={self.generator.x_input_placeholder: generator_x_batch})

                self.generator_train_writer.add_summary(summary, i)

                if i % 10 == 0:
                    loss_discriminator_from_generator_input = sess.run(self.discriminator_of_generator.cost,
                                                                       feed_dict={self.generator.x_input_placeholder:
                                                                                      generator_x_batch})

                    loss_discriminator_from_real_input = sess.run(self.discriminator.cost,
                                                                  feed_dict={self.discriminator.input_placeholder:
                                                                                 discriminator_x_batch})

                    loss_generator = sess.run(self.discriminator_of_generator.generator_cost,
                                              feed_dict={self.generator.x_input_placeholder: generator_x_batch})

                    print("Iteration number %d, discriminator_loss: %.8f, generator_loss: %.8f" %
                          (i, loss_discriminator_from_generator_input + loss_discriminator_from_real_input,
                           loss_generator))

                    # Save the model :
                    save_path = self.saver.save(sess, os.path.join(self.checkpoint_dir, 'iter_' + str(i) + '.ckpt'))
                    print("Model saved in file: %s" % save_path)

                if i % 100 == 0:
                    if self.is_input_an_image:
                        # sample from the generator and plot :
                        y_samples = discriminator_y_train[:64]
                        random_samples = np.random.uniform(-1, 1, size=(64, self.generator.input_dim))
                        output_images = self.generator.run_generator(sess, random_samples, y_samples)

                        manifold_h = int(np.floor(np.sqrt(64)))
                        manifold_w = int(np.ceil(np.sqrt(64)))
                        assert manifold_h * manifold_w == 64
                        size = manifold_h, manifold_w

                        self.save_batch_of_images(output_images, size, os.path.join(self.generated_samples_dir,
                                                                                    'sample_iter_' + str(i) + '.png'))

                    else:
                        # sample from the generator and plot :
                        random_samples = np.random.uniform(-1, 1, size=(5, self.generator.input_dim))
                        output_generator = self.generator.run_generator(sess, random_samples)
                        for j, e in enumerate(output_generator):
                            print(e)
                            plt.figure()
                            plt.plot(e)
                            plt.savefig(os.path.join(self.generated_samples_dir, str(i) + str(j) + "_ecg_gan.png"))
                            # plt.show()


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

    @staticmethod
    def save_batch_of_images(images, size, image_path):
        """
        given a batch of images saves to one image marging all to one.
        :param images: [batch size, h, w, c]
        :param size:
        :param image_path: path to save
        :return:
        """

        images = (images + 1.) / 2.
        h, w = images.shape[1], images.shape[2]
        if images.shape[3] in (3, 4):
            c = images.shape[3]
            img = np.zeros((h * size[0], w * size[1], c))
            for idx, image in enumerate(images):
                i = idx % size[1]
                j = idx // size[1]
                img[j * h:j * h + h, i * w:i * w + w, :] = image
            manipulated_img = img
        elif images.shape[3] == 1:
            img = np.zeros((h * size[0], w * size[1]))
            for idx, image in enumerate(images):
                i = idx % size[1]
                j = idx // size[1]
                img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
            manipulated_img = img
        else:
            raise ValueError('in merge(images,size) images parameter '
                             'must have dimensions: HxW or HxWx3 or HxWx4')

        manipulated_img = np.squeeze(manipulated_img)

        return scipy.misc.imsave(image_path, manipulated_img)

    def load_checkpoint(self):
        """
        Loads a model that was already trained with a checkpoint file
        :return:
        """
        # TODO: complete this .....
        print(" [*] Reading checkpoints...")
        checkpoint_dir = self.checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            self.saver.restore(self.session, os.path.join(checkpoint_dir, ckpt_name))

            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


# Helper functions for celebA data:
def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    """
    Opens a jpg file into an array and also might resize the image to desired shape.
    :param image_path:
    :param input_height:
    :param input_width:
    :param resize_height:
    :param resize_width:
    :param crop:
    :param grayscale:
    :return:
    """
    image = scipy.misc.imread(image_path).astype(np.float)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)


def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    """
    transforms an image to desired shape.
    :param image:
    :param input_height:
    :param input_width:
    :param resize_height:
    :param resize_width:
    :param crop:
    :return:
    """
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize( x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])


if __name__ == "__main__":

    print("Testing GAN on celcbA images:")
    mnist_gan = GenericGAN(discriminator_input_height=28, discriminator_input_width=28,
                        discriminator_number_of_input_channels=1, discriminator_number_of_filters_at_first_layer=64,
                        discriminator_arch_file=None, generator_input_dim=100, generator_output_height=28,
                        generator_output_width=28, generator_output_channels=1, generator_architecture_file=None,
                        tensor_board_logs_dir='./logs/dcgan/mnist', is_input_an_image=True,
                           checkpoint_dir='./saved_models/dcgan/mnist', generated_samples_dir='samples/dcgan/mnist',
                        y_labels_dim=10, with_activation_function_on_last_layer_of_generator=True,
                           gan_type='vanilla_conditional_gan')

    # load the mnist data :
    mnist_data = input_data.read_data_sets("mnist_data", one_hot=True)
    train_set = mnist_data.train
    x = train_set.images
    x = x.reshape((55000, 28, 28, 1)).astype(np.float32)
    y_tags = train_set.labels

    mnist_gan.train(discriminator_x_train=x, discriminator_y_train=y_tags, num_of_iterations=1000000, batch_size=128,
                    learning_rate=0.0002, momentum_term_adam=0.5)


    '''
    print("Testing GAN on celcbA images:")
    # load the celebA data :
    celebA_list_of_images_names = glob(os.path.join("./data", 'celebA', '*.jpg'))
    # For debugging - check image dimensions and print them:
    imreadImg = scipy.misc.imread(celebA_list_of_images_names[0]).astype(np.float)
    print("image dims:", imreadImg.shape)
    print("Number of images: ", len(celebA_list_of_images_names))
    # Crop the images
    '''
    '''
    train_images = [get_image(img_file,
                              input_height=108,
                              input_width=108,
                              resize_height=64,
                              resize_width=64,
                              crop=True,
                              grayscale=False) for img_file in celebA_list_of_images_names]
    '''
    '''
    # Test on celebA:
    celebA_gan = GenericGAN(discriminator_input_height=64, discriminator_input_width=64,
                           discriminator_number_of_input_channels=3, discriminator_number_of_filters_at_first_layer=64,
                           discriminator_arch_file=None, generator_input_dim=100, generator_output_height=64,
                           generator_output_width=64, generator_output_channels=3, generator_architecture_file=None,
                           tensor_board_logs_dir='./logs/celebA', is_input_an_image=True,
                            checkpoint_dir='./saved_models/dcgan/celebA',
                            generated_samples_dir='samples/dcgan/celebA')

    celebA_gan.train(discriminator_x_train=np.array(celebA_list_of_images_names), discriminator_y_train=np.array(celebA_list_of_images_names), num_of_iterations=1000000,
                     batch_size=128, learning_rate=0.0002, momentum_term_adam=0.5, crop_images=True)

    '''







