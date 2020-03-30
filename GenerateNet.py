# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data:
from utils import *
import tensorflow as tf
import numpy as np
from params import *
from HyperNet import genWeight
from loadData import DataSet
import random


class GenNet(object):

    def __init__(self, category='Mnist', vis_step=10, Train_Epochs=200, z_batch_size=128, image_batch_size=128, z_size=100, lr=0.001,
                 history_dir='./', checkpoint_dir='./', logs_dir='./', gen_dir='./', test_dir='./'):
        self.test = False

        self.category = category
        self.epoch = Train_Epochs
        self.img_size = 28 if (category == 'Fashion-Mnist' or category == 'Mnist') else 64
        self.z_batch_size = z_batch_size
        self.image_batch_size = image_batch_size
        self.batch_size = image_batch_size*z_batch_size
        self.z_size = z_size
        self.vis_step = vis_step

        self.lr = lr
        self.channel = 1 if (category == 'Fashion-Mnist' or category == 'Mnist') else 3
        self.history_dir = history_dir
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.gen_dir = gen_dir
        self.test_dir = test_dir

        self.z_i = tf.placeholder(tf.float32, shape=[self.z_batch_size, self.image_batch_size, self.z_size], name='latent_img')
        self.z_h = tf.placeholder(tf.float32, shape=[self.z_batch_size, self.z_size], name='latent_hyper')

        self.x = tf.placeholder(tf.float32, shape=[self.image_batch_size*self.z_batch_size, self.img_size, self.img_size, self.channel],
                                name='image')


    def build_Model(self):
        self.weights = self.HyperNet(self.z_h, reuse=False)
        self.weights_test = self.HyperNet(self.z_h, reuse=True)

        self.gen = self.Generator(self.z_i, self.weights, reuse=False)
        self.gen_test = self.Generator(self.z_i, self.weights_test, reuse=True)

        self.D_x = self.Discriminator(self.x, reuse=False)
        self.D_gen = self.Discriminator(self.gen, reuse=True)

        """
        Loss and Optimizer
        """
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_x, labels=tf.ones_like(self.D_x)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_gen, labels=tf.zeros_like(self.D_gen)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_gen, labels=tf.ones_like(self.D_gen)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_var = [var for var in tf.trainable_variables() if var.name.startswith('Gen')]
        self.d_var = [var for var in tf.trainable_variables() if var.name.startswith('Dis')]

        self.d_optim = tf.train.AdamOptimizer(self.lr) \
            .minimize(self.d_loss, var_list=self.d_var)
        self.g_optim = tf.train.AdamOptimizer(self.lr) \
            .minimize(self.g_loss, var_list=self.g_var)

        """
        Logs
        """
        tf.summary.scalar('g_loss', tf.reduce_mean(self.g_loss))
        tf.summary.scalar('d_loss', tf.reduce_mean(self.d_loss))
        # TODO showing specifically
        # tf.summary.histogram('hyper params', self.hyper_var)
        self.summary_op = tf.summary.merge_all()

    def Generator(self, z, weights, reuse=False):
        with tf.variable_scope('Gen', reuse=reuse):
            params = GeneratorParams(category=self.category, img_size=self.img_size, z_size=self.z_size)

            if self.category == 'Fashion-Mnist' or self.category == 'Mnist':
                fc1_w = weights['fc1_w']
                fc1_b = weights['fc1_b']
                fc2_w = weights['fc2_w']
                fc2_b = weights['fc2_b']

                dc1_w = weights['dc1_w']
                dc1_b = weights['dc1_b']
                dc2_w = weights['dc2_w']
                dc2_b = weights['dc2_b']
                """
                fc1
                """

                fc1 = tf.add(tf.reduce_sum(tf.expand_dims(z, -1) * tf.expand_dims(fc1_w, 1), axis=2), tf.expand_dims(fc1_b, 1), name='fc1')
                fc1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(fc1, is_training=True))


                """
                fc2
                """
                fc2 = tf.add(tf.reduce_sum(tf.expand_dims(fc1, -1) * tf.expand_dims(fc2_w, 1), axis=2), tf.expand_dims(fc2_b,1), name='fc2')
                fc2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(fc2, is_training=True))

                fc2 = tf.reshape(fc2, [self.z_batch_size, self.image_batch_size, 7, 7, -1])

                """
                dc1
                """
                output_shape = getOutputShape(
                    (self.image_batch_size, params.dc1_img_size, params.dc1_img_size, params.dc1_size))

                fn = lambda u: tf.add(
                    tf.nn.conv2d_transpose(value=u[0], filter=u[1], output_shape=output_shape, strides=[1, 2, 2, 1],
                                           padding='SAME'), u[2])
                dc1 = tf.map_fn(fn, elems=[fc2, dc1_w, dc1_b], dtype=tf.float32, name='dc1')
                dc1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(dc1, is_training=True))


                """
                dc2
                """
                output_shape = getOutputShape(
                    (self.image_batch_size, params.dc2_img_size, params.dc2_img_size, params.dc2_size))

                fn = lambda u: tf.add(
                    tf.nn.conv2d_transpose(value=u[0], filter=u[1], output_shape=output_shape, strides=[1, 2, 2, 1],
                                           padding='SAME'), u[2])
                dc2 = tf.map_fn(fn, elems=[dc1, dc2_w, dc2_b], dtype=tf.float32, name='dc2')

                output = tf.nn.tanh(dc2)
                output = tf.reshape(output, [-1, 28, 28, 1])

        return output

    def HyperNet(self, z, reuse=False):
        Hpyer_params = HyperNetParams(category=self.category, img_size=self.img_size, z_size=self.z_size)
        Gen_params = GeneratorParams(category=self.category, img_size=self.img_size, z_size=self.z_size)
        weights = {}
        with tf.variable_scope('Gen', reuse=reuse):
            if self.category == 'Mnist':
                # [1, z] - [1, 300] - [1, 300] - [1, prod+prod]
                with tf.variable_scope('code'):
                    weightSize = [Hpyer_params.extractor_w1, Hpyer_params.extractor_w2, Hpyer_params.extractor_w3]
                    codes = genWeight(layerSize=Hpyer_params.extractor_hiddenlayer_size, weightSize=weightSize,
                                      input=z)
                    # code1 128, 1024, 15
                    startIdx = 0
                    endIdx = startIdx + np.prod(Hpyer_params.code1_size)
                    code1 = tf.identity(tf.reshape(codes[:, startIdx:endIdx], [-1, ] + Hpyer_params.code1_size),
                                        name='code1')

                    # code2 128, 6272, 15
                    startIdx = endIdx
                    endIdx = startIdx + np.prod(Hpyer_params.code2_size)
                    code2 = tf.identity(
                        tf.reshape(codes[:, startIdx:endIdx], [-1, ] + Hpyer_params.code2_size, name='code2'))

                    # code3 128, 128, 15
                    startIdx = endIdx
                    endIdx = startIdx + np.prod(Hpyer_params.code3_size)
                    code3 = tf.identity(
                        tf.reshape(codes[:, startIdx:endIdx], [-1, ] + Hpyer_params.code3_size, name='code3'))

                    # code4 128, 1, 15
                    startIdx = endIdx
                    endIdx = startIdx + np.prod(Hpyer_params.code4_size)
                    code4 = tf.identity(
                        tf.reshape(codes[:, startIdx:endIdx], [-1, ] + Hpyer_params.code4_size, name='code4'))

                # gen fc1 filter code[128, 1024 15] - [128, 1024, 40] - [128, 1024, 40] - [128, 1024, z+10+1] - [128, z+10+1, 1024]
                with tf.variable_scope('fc1_w', reuse=reuse):
                    weightSize = [Hpyer_params.fc1_w1_size, Hpyer_params.fc1_w2_size, Hpyer_params.fc1_w3_size]
                    w_b = genWeight(layerSize=Hpyer_params.w_gen_hiddenlayer_size, weightSize=weightSize,
                                    input=code1)
                    w = tf.identity(
                        tf.reshape(w_b[:, :, :Gen_params.fc1_filter_size[0]], [-1, ] + Gen_params.fc1_filter_size),
                        name='reshaped_w1')
                    b = tf.identity(tf.reshape(w_b[:, :, Gen_params.fc1_filter_size[0]:],
                                               [-1, ] + [Gen_params.fc1_filter_size[1]]), name='reshaped_b1')
                    weights['fc1_w'] = w
                    weights['fc1_b'] = b

                # gen fc2 filter code[128, 6272 15] - [128, 6272, 40] - [128, 6272, 40] - [128, 6272, 1024+10+1] - [128, 1024+10+1, 6272]
                with tf.variable_scope('fc2_w', reuse=reuse):
                    weightSize = [Hpyer_params.fc2_w1_size, Hpyer_params.fc2_w2_size, Hpyer_params.fc2_w3_size]
                    w_b = genWeight(layerSize=Hpyer_params.w_gen_hiddenlayer_size, weightSize=weightSize,
                                    input=code2)
                    w = tf.identity(
                        tf.reshape(w_b[:, :, :Gen_params.fc2_filter_size[0]], [-1, ] + Gen_params.fc2_filter_size),
                        name='reshaped_w1')
                    b = tf.identity(tf.reshape(w_b[:, :, Gen_params.fc2_filter_size[0]:],
                                               [-1, ] + [Gen_params.fc2_filter_size[1]]), name='reshaped_b1')
                    weights['fc2_w'] = w
                    weights['fc2_b'] = b

                # gen fc2 filter code[128, 128, 15] - [128, 128, 40] - [128, 128, 40] - [128, 128, 25*42+1] - [128, 5, 5, 42, 128]
                with tf.variable_scope('dc1_w', reuse=reuse):
                    weightSize = [Hpyer_params.dc1_w1_size, Hpyer_params.dc1_w2_size, Hpyer_params.dc1_w3_size]
                    w_b = genWeight(layerSize=Hpyer_params.w_gen_hiddenlayer_size, weightSize=weightSize,
                                    input=code3)
                    w = tf.identity(
                        tf.reshape(w_b[:, :, :Gen_params.dc1_filter_size ** 2 * (Gen_params.fc2_filter_size[1] // 49)],
                                   [-1, ] + [Gen_params.dc1_filter_size, Gen_params.dc1_filter_size,
                                             Gen_params.dc1_size, Gen_params.fc2_filter_size[1] // 49]),
                        name='reshaped_w2')
                    b = tf.identity(
                        tf.reshape(w_b[:, :, Gen_params.dc1_filter_size ** 2 * (Gen_params.fc2_filter_size[1] // 49):],
                                   [-1, ] + [Gen_params.dc1_size]), name='reshaped_b2')
                    weights['dc1_w'] = w
                    weights['dc1_b'] = b

                # gen fc2 filter code[128, 1, 15] - [128, 1, 40] - [128, 1, 40] - [128, 1, 25*138+1] - [128, 5, 5, 138, 1]
                with tf.variable_scope('dc2_w', reuse=reuse):
                    weightSize = [Hpyer_params.dc2_w1_size, Hpyer_params.dc2_w2_size, Hpyer_params.dc2_w3_size]
                    w_b = genWeight(layerSize=Hpyer_params.w_gen_hiddenlayer_size, weightSize=weightSize,
                                    input=code4)
                    w = tf.identity(tf.reshape(w_b[:, :, :Gen_params.dc2_filter_size ** 2 * (Gen_params.dc1_size)],
                                               [-1, ] + [Gen_params.dc2_filter_size, Gen_params.dc2_filter_size,
                                                         Gen_params.dc2_size, Gen_params.dc1_size]), name='reshaped_w2')
                    b = tf.identity(tf.reshape(w_b[:, :, Gen_params.dc2_filter_size ** 2 * (Gen_params.dc1_size):],
                                               [-1, ] + [Gen_params.dc2_size]), name='reshaped_b2')
                    weights['dc2_w'] = w
                    weights['dc2_b'] = b

        return weights

    def Discriminator(self, x, reuse=False):
        with tf.variable_scope('Dis', reuse=reuse):
            if self.category == 'Fashion-Mnist' or self.category == 'Mnist':
                c1 = tf.nn.leaky_relu(conv2d(x, 1, name='c1'))

                c2 = tf.contrib.layers.batch_norm(conv2d(c1, 64, name='c2'), is_training=True)
                c2 = tf.nn.leaky_relu(c2)
                c2 = tf.reshape(c2, [self.image_batch_size*self.z_batch_size, -1])

                fc1 = tf.layers.dense(inputs=c2, units=1024, name='fc1')
                fc1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(fc1, is_training=True))

                output = tf.layers.dense(inputs=fc1, units=1, name='fc2')

            return output

    def train(self, sess):
        self.build_Model()

        data = DataSet(img_size=self.img_size, batch_size=self.batch_size, category=self.category)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=10)

        writer = tf.summary.FileWriter(self.logs_dir, sess.graph)

        start = 0
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)

        if latest_checkpoint:
            latest_checkpoint.split('-')
            start = int(latest_checkpoint.split('-')[-1])
            saver.restore(sess, latest_checkpoint)
            print('Loading checkpoint {}.'.format(latest_checkpoint))

        tf.get_default_graph().finalize()

        # latent_gen = np.random.normal(size=(len(data), self.z_size))

        for epoch in range(start + 1, self.epoch):
            num_batch = int(len(data) / self.batch_size)
            d_losses = []
            g_losses = []
            for step in range(num_batch):
                obs, labels = data.NextBatch(step)
                # z = latent_gen[step * self.batch_size: (step + 1) * self.batch_size].copy()
                z_i = np.random.normal(size=(self.z_batch_size, self.image_batch_size, self.z_size))
                z_h = np.random.normal(size=(self.z_batch_size, self.z_size))


                d_loss, _ = sess.run([self.d_loss, self.d_optim], feed_dict={self.z_i: z_i, self.x: obs, self.z_h:z_h})
                d_losses.append(d_loss)

                g_loss, _ = sess.run([self.g_loss, self.g_optim], feed_dict={self.z_i: z_i, self.z_h:z_h, self.x:obs})
                g_losses.append(g_loss)
                # writer.add_summary(summary, global_step=epoch)

                _ = sess.run(self.g_optim, feed_dict={self.z_i: z_i, self.z_h:z_h, self.x:obs})

            print(epoch, " dis Loss: ", np.mean(d_losses), " gen loss: ", np.mean(g_losses))
            if epoch % self.vis_step == 0:
                self.visualize(saver, sess, len(data), epoch, np.random.normal(size=(len(data), self.z_size)), data)
        self.visualize_test(sess)

    def visualize(self, saver, sess, num_data, epoch, latent_gen, data):
        saver.save(sess, "%s/%s" % (self.checkpoint_dir, 'model.ckpt'), global_step=epoch)
        # z = latent_gen[idx * self.batch_size: (idx + 1) * self.batch_size]

        """
        Generation
        """

        z_i = np.random.normal(size=(self.z_batch_size, self.image_batch_size, self.z_size))
        z_h = np.random.normal(size=(self.z_batch_size, self.z_size))

        sys = sess.run(self.gen, feed_dict={self.z_i: z_i, self.z_h: z_h})
        sys = np.array((sys + 1) * 127.5, dtype=np.float)
        path = self.gen_dir + 'epoch' + str(epoch) + 'gens.jpg'
        show_in_one(path, sys, column=16, row=8)

    def visualize_test(self, sess):

        """
        :param sess:
        :return:
        """

        """
            test a
        """
        z_h = np.random.normal(size=(self.z_batch_size, self.z_size))
        z_i = np.random.normal(size=(self.z_batch_size, self.image_batch_size, self.z_size))
        sys = sess.run(self.gen, feed_dict={self.z_i: z_i, self.z_h:z_h})
        sys = np.array((sys + 1) * 127.5, dtype=np.float)
        path = self.test_dir + 'diff_weights_same_z' + '1.jpg'
        show_in_one(path, sys, column=16, row=8)

        """
            test b
        """
        z_h = np.random.normal(size=(self.z_batch_size, self.z_size))
        # z_i = np.random.normal(size=(self.z_batch_size, self.image_batch_size, self.z_size))
        sys = sess.run(self.gen, feed_dict={self.z_i: z_i, self.z_h:z_h})
        sys = np.array((sys + 1) * 127.5, dtype=np.float)
        path = self.test_dir + 'diff_weights_same_z' + '2.jpg'
        show_in_one(path, sys, column=16, row=8)

        """
            test c
        """
        z_h = np.random.normal(size=(self.z_batch_size, self.z_size))
        z_i = np.random.normal(size=(self.z_batch_size, self.image_batch_size, self.z_size))
        sys = sess.run(self.gen, feed_dict={self.z_i: z_i, self.z_h:z_h})
        sys = np.array((sys + 1) * 127.5, dtype=np.float)
        path = self.test_dir + 'diff_z_same_weight' + '1.jpg'
        show_in_one(path, sys, column=16, row=8)

        """
            test d
        """
        # z_h = np.random.normal(size=(self.z_batch_size, self.z_size))
        z_i = np.random.normal(size=(self.z_batch_size, self.image_batch_size, self.z_size))
        sys = sess.run(self.gen, feed_dict={self.z_i: z_i, self.z_h:z_h})
        sys = np.array((sys + 1) * 127.5, dtype=np.float)
        path = self.test_dir + 'diff_z_same_weight' + '2.jpg'
        show_in_one(path, sys, column=16, row=8)
