import os
import sys
import time
from itertools import count
import math
import random
import tensorflow as tf
from keras.utils.generic_utils import Progbar
from .nets import resnet50, eigen_coarse, eigen_fine
from .data_utils import DirectoryIterator

class Learner(object):
    def __init__(self):
        pass

    def read_from_disk(self, inputs_queue):
        """Consumes the inputs queue.
        Args:
            filename_and_label_tensor: A scalar string tensor.
        Returns:
            Two tensors: the decoded images, and the labels.
        """
        rgb_path = inputs_queue[0][0]
        depth_path = inputs_queue[0][1]

        file_rgb = tf.read_file(rgb_path)
        file_depth = tf.read_file(depth_path)

        rgb_seq = tf.image.decode_image(file_rgb, channels=3)
        rgb_seq.set_shape([None, None, 3])
        depth_seq = tf.image.decode_png(file_depth, channels=1)

        return rgb_seq, depth_seq

    def preprocess_rgb(self, rgb, is_training):
        aspect_ratio = tf.constant(self.config['output_width']/self.config['output_height'],
                                    dtype=tf.float32)
        data_shape = tf.to_float(tf.shape(rgb))
        [data_height, data_width] = [data_shape[0], data_shape[1]]

        rgb = tf.image.per_image_standardization(rgb)
        # Data augmentation
        rgb = tf.cond(is_training,
                      lambda: tf.image.random_saturation(rgb, 0.7, 1.3, seed=1), lambda: rgb)
        rgb = tf.cond(is_training,
                      lambda: tf.image.random_flip_left_right(rgb, seed=1), lambda: rgb)
        # Crop images to comply with network input
        rgb = tf.cond(data_width > aspect_ratio*data_width,
            lambda: tf.image.resize_image_with_crop_or_pad(rgb,
                tf.to_int32(data_height), tf.to_int32(aspect_ratio*data_height)),
            lambda: tf.image.resize_image_with_crop_or_pad(rgb,
                tf.to_int32(data_height/aspect_ratio), tf.to_int32(data_width)))
        # Resize images to comply with network input/output
        rgb = tf.image.resize_images(rgb,
            [self.config['input_height'], self.config['input_width']])
        # Convert uint8 to float32
        rgb = tf.cast(rgb, dtype=tf.float32)/255.

        return rgb

    def preprocess_depth(self, depth, is_training):
        # Data augmentation
        depth = tf.cond(is_training,
                        lambda: tf.image.random_flip_left_right(depth, seed=1),
                        lambda: depth)
        aspect_ratio = tf.constant(self.config['output_width']/self.config['output_height'],
                                    dtype=tf.float32)
        data_shape = tf.to_float(tf.shape(depth))
        [data_height, data_width] = [data_shape[0], data_shape[1]]

        # Crop depth images to comply with network input
        depth = tf.cond(data_width > (aspect_ratio*data_width),
            lambda: tf.image.resize_image_with_crop_or_pad(depth,
                tf.to_int32(data_height), tf.to_int32(aspect_ratio*data_height)),
            lambda: tf.image.resize_image_with_crop_or_pad(depth,
                tf.to_int32(data_height/aspect_ratio), tf.to_int32(data_width)))
        # Resize images to comply with network input/output
        depth = tf.image.resize_images(depth,
                [self.config['output_height'], self.config['output_width']])
        # Convert uint8 to float32
        depth = tf.cast(depth, dtype=tf.float32)/255.

        return depth

    def generate_batches(self, data_dir, is_training):
        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list = self.get_filenames_list(data_dir)
        inputs_queue = tf.train.slice_input_producer([file_list], seed=seed, shuffle=True)
        rgb_seq, depth_seq = self.read_from_disk(inputs_queue)
        # Resize images to target size and preprocess them
        rgb_seq = self.preprocess_rgb(rgb_seq, is_training)
        depth_seq = self.preprocess_depth(depth_seq, is_training)
        # Form training batches
        rgb_batch, depth_batch = tf.train.batch([rgb_seq,
             depth_seq],
             batch_size=self.config['batch_size'],
             num_threads=self.config['num_threads'],
             capacity=self.config['capacity_queue'],
             allow_smaller_final_batch=True)
        return [rgb_batch, depth_batch], len(file_list)

    def get_filenames_list(self, directory):
        """ This function should return all the filenames of the
            files you want to train on.
            In case of classification, it should also return labels.

            Args:
                directory: dataset directory
            Returns:
                List of filenames, [List of associated labels]
        """
        iterator = DirectoryIterator(directory, dataset=self.config['ds'],
                                    follow_links=True)

        return iterator.filenames

    def loss_mse_scale_invariant(self, prediction, target, alpha=0.5):
        """ This function should compute the loss as proposed by Eigen
        Args:
            prediction: A batch of depth predictions
            target: A batch of groun truth depth images
            alpha: A factor to specifiy if the loss is entirely scale loss
                (alpha = 1.0) or regular mse (alpha = 0.0)
        Returns:
            loss: Scalar loss
        """
        diff = prediction - target
        msd = tf.reduce_mean(tf.square(diff))
        smd = tf.square(tf.reduce_mean(diff))
        loss = msd - alpha*smd
        return loss

    def save(self, sess, directory, step):
        model_name = 'model'
        checkpoint_dir = os.path.join(directory, self.config['name'])
        print(" [*] Saving checkpoint to {}/model-{}".format(checkpoint_dir,
                                                             step))
        if step == 'best':
            self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name + '.best'))
        else:
            self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def setup_inference(self, config):
        """Sets up the inference graph.
        Args:
            config: config dictionary.
        """
        self.config = config
        self.build_test_graph()

    def inference(self, inputs, sess):
        """Outputs a dictionary with the results of the required operations.
        Args:
            inputs: Dictionary with variable to be feed to placeholders
            sess: current session
        Returns:
            results: dictionary with output of testing operations.
        """
        if 'depth' in inputs.keys():
            fetches = {"loss": self.test_loss,
                       "depth_prediction": self.test_prediction}
            feed_dict = {self.input_rgb: inputs['rgb'],
                         self.depth_labels: inputs['depth']}
        else:
            fetches = {"depth_prediction": self.test_prediction}
            feed_dict = {self.input_rgb: inputs['rgb']}

        results = {}
        results = sess.run(fetches, feed_dict)
        return results


class LearnerEigen(Learner):
    def __init__(self):
        Learner.__init__(self)

    def build_train_graph(self):
        is_training_ph = tf.placeholder(tf.bool, shape=(), name="is_training")
        fine_tuning_ph = tf.placeholder(tf.bool, shape=(), name="fine_tuning")

        with tf.name_scope("data_loading"):
            print('Parsing training directory...')
            train_batch, n_samples_train = self.generate_batches(
                    self.config['train_dir'], is_training=tf.constant(True))
            print('Parsing validation directory...')
            val_batch, n_samples_test = self.generate_batches(
                    self.config['val_dir'], is_training=tf.constant(False))
            current_batch = tf.cond(is_training_ph, lambda: train_batch,
                                                    lambda: val_batch)
            image_batch, depth_labels = current_batch[0], current_batch[1]

        with tf.name_scope("coarse_prediction"):
            # predict coarse depth map from RGB input image
            depth_coarse = eigen_coarse(image_batch,
                                        is_training=True,
                                        l2_reg_scale=self.config['l2_reg_scale'])
        with tf.name_scope("coarse_loss"):
            coarse_loss = self.loss_mse_scale_invariant(depth_coarse, depth_labels,
                                                        alpha=0.5)

        with tf.name_scope("fine_prediction"):
            # predict fine depth map from coarse depth and RGB input image
            depth_fine = eigen_fine(image_batch, depth_coarse,
                                    is_training=True,
                                    l2_reg_scale=self.config['l2_reg_scale'])
        with tf.name_scope("fine_loss"):
            fine_loss = self.loss_mse_scale_invariant(depth_fine, depth_labels,
                                                      alpha=0.5)

        with tf.name_scope("train_op"):
            reg_losses = tf.reduce_sum(
                          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            train_vars_coarse = [var for var in tf.trainable_variables(
                                            scope="coarse_prediction")]
            train_vars_fine = [var for var in tf.trainable_variables(
                                            scope="fine_prediction")]
            optimizer_coarse  = tf.train.AdamOptimizer(
                    self.config['learning_rate_coarse'], self.config['beta1'])
            optimizer_fine  = tf.train.AdamOptimizer(
                    self.config['learning_rate_fine'], self.config['beta1'])
            self.grads_and_vars_coarse = optimizer_coarse.compute_gradients(
                    coarse_loss+reg_losses, var_list=train_vars_coarse)
            self.grads_and_vars_fine = optimizer_fine.compute_gradients(
                    fine_loss+reg_losses, var_list=train_vars_fine)
            #Select train operation depening on train mode coarse/fine
            self.train_op = tf.cond(fine_tuning_ph,
                    lambda:optimizer_fine.apply_gradients(self.grads_and_vars_fine),
                    lambda:optimizer_coarse.apply_gradients(self.grads_and_vars_coarse))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.incr_global_step = tf.assign(self.global_step, self.global_step+1)

        self.coarse_depth_prediction = depth_coarse
        self.fine_depth_prediction = depth_fine
        self.depth_labels = depth_labels
        self.input_img = image_batch
        self.coarse_loss = coarse_loss
        self.fine_loss = fine_loss
        self.loss = tf.cond(fine_tuning_ph, lambda:fine_loss, lambda:coarse_loss)
        self.train_steps_per_epoch = int(math.ceil(n_samples_train
                                            /self.config['batch_size']))
        self.val_steps_per_epoch = int(math.ceil(n_samples_test
                                            /self.config['batch_size']))
        self.is_training = is_training_ph
        self.fine_tuning = fine_tuning_ph

    def collect_summaries(self):
        """Collects all summaries to be shown in the tensorboard"""
        tf.summary.scalar("Train Loss", self.loss,
                collections=["step_sum_coarse", "step_sum_fine"])
        tf.summary.image("Train Coarse Depth Prediction", self.coarse_depth_prediction,
                collections=["step_sum_coarse", "step_sum_fine"], max_outputs=3)
        tf.summary.image("Train Depth Prediction", self.fine_depth_prediction,
                collections=["step_sum_fine"], max_outputs=3)
        tf.summary.image("Train GT Depth Image", self.depth_labels,
                collections=["step_sum_coarse", "step_sum_fine"], max_outputs=3)
        tf.summary.image("Train Input RGB Image", self.input_img,
                collections=["step_sum_coarse", "step_sum_fine"], max_outputs=3)

        self.step_sum_coarse = tf.summary.merge([tf.get_collection('step_sum'),
            tf.get_collection('step_sum_coarse')])
        self.step_sum_fine = tf.summary.merge([tf.get_collection('step_sum'),
            tf.get_collection('step_sum_fine')])

        tf.summary.image("Validation Depth Prediction", self.fine_depth_prediction,
                collections=["val_sum_fine"], max_outputs=3)
        tf.summary.image("Validation Depth Prediction", self.coarse_depth_prediction,
                collections=["val_sum_coarse"], max_outputs=3)
        tf.summary.image("Validation GT Depth Image", self.depth_labels,
                collections=["val_sum_coarse", "val_sum_fine"], max_outputs=3)
        tf.summary.image("Validation Input RGB Image", self.input_img,
                collections=["val_sum_coarse", "val_sum_fine"], max_outputs=3)
        self.validation_loss = tf.placeholder(tf.float32, [])
        tf.summary.scalar("Validation Loss Coarse", self.validation_loss,
                          collections = ["val_sum_coarse"])
        tf.summary.scalar("Validation Loss Fine", self.validation_loss,
                          collections = ["val_sum_fine"])
        self.val_sum_coarse = tf.summary.merge(tf.get_collection('val_sum_coarse'))
        self.val_sum_fine = tf.summary.merge(tf.get_collection('val_sum_fine'))

    def save(self, sess, directory, step):
        model_name = 'model'
        checkpoint_dir = os.path.join(directory, self.config['name'])
        print(" [*] Saving checkpoint to {}/model-{}".format(checkpoint_dir, step))
        if step == 'best':
            self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name + '.best'))
        elif step == 'switch':
            self.saver_coarse.save(sess,
                        os.path.join(checkpoint_dir, model_name + '.switch'))
        else:
            self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def train(self, config):
        """High level train function.
        Args:
            config: Configuration dictionary
        Returns:
            None
        """
        self.config = config
        self.build_train_graph()
        self.collect_summaries()
        self.min_val_loss = float('inf') # Initialize to max value
        parameter_count_coarse = tf.reduce_sum(
                [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables(
                    scope="coarse_prediction")])
        parameter_count_fine = tf.reduce_sum(
                [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables(
                    scope="fine_prediction")])
        self.saver = tf.train.Saver([var for var in \
            tf.trainable_variables()] +  [self.global_step], max_to_keep=3)
        self.saver_coarse = tf.train.Saver([var for var in \
            tf.trainable_variables()] +  [self.global_step], max_to_keep=1)
        sv = tf.train.Supervisor(logdir=os.path.join(config['log_dir'],config['name']),
                                 save_summaries_secs=0,
                                 saver=None)
        with sv.managed_session() as sess:
            print("Number of trainable params (Coarse): {}".format(
                sess.run(parameter_count_coarse)))
            print("Number of trainable params (Fine): {}".format(
                sess.run(parameter_count_fine)))
            if config['resume_train']:
                print("Resume training from previous checkpoint")
                checkpoint = tf.train.latest_checkpoint(config['resume_weights'])
                self.saver.restore(sess, checkpoint)

            progbar = Progbar(target=self.train_steps_per_epoch)

            # start training by optimizing only the coarse net
            # training loop
            self.do_fine_tuning = False
            for step in count(start=1):
                if sv.should_stop():
                    break
                start_time = time.time()
                # Variables that are fetched every step
                fetches = { "train" : self.train_op,
                            "global_step" : self.global_step,
                            "incr_global_step": self.incr_global_step}
                # Variables that are only fetched when summary is saved
                if step % config['summary_freq'] == 0:
                    if self.do_fine_tuning:
                        fetches["loss"] = self.fine_loss
                        fetches["summary"] = self.step_sum_fine
                    else:
                        fetches["loss"] = self.coarse_loss
                        fetches["summary"] = self.step_sum_coarse

                # Runs the series of operations
                results = sess.run(fetches,
                        feed_dict={ self.is_training: True,
                                    self.fine_tuning: self.do_fine_tuning})

                gs = results["global_step"]
                if step % config['summary_freq'] == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)

                if step % self.train_steps_per_epoch == 0:
                    # This differ from the last when resuming training
                    progbar.update(self.train_steps_per_epoch)
                    train_epoch = int(gs / self.train_steps_per_epoch)
                    self.epoch_end_callback(sess, sv, train_epoch, config)
                    progbar = Progbar(target=self.train_steps_per_epoch)
                    if (train_epoch == self.config['max_epochs']):
                        print("-------------------------------")
                        print("Training completed successfully")
                        print("-------------------------------")
                        break
                else:
                    progbar.update(step % self.train_steps_per_epoch)

    def epoch_end_callback(self, sess, sv, epoch_num, config):
        # Check epoch number to switch from coarse to fine network
        if self.do_fine_tuning == True:
            fetches = { "loss" : self.coarse_loss }
        else:
            fetches = { "loss" : self.fine_loss }

        val_loss = 0
        for i in range(self.val_steps_per_epoch):
            results = sess.run(fetches,
                        feed_dict={ self.is_training: False,
                                    self.fine_tuning: self.do_fine_tuning})
            val_loss += results['loss']
        val_loss = val_loss / self.val_steps_per_epoch
        # Log to Tensorflow board
        if self.do_fine_tuning == True:
            val_sum = sess.run(self.val_sum_fine,
                               feed_dict ={self.is_training: False,
                                           self.validation_loss: val_loss})
        else:
            val_sum = sess.run(self.val_sum_coarse,
                               feed_dict ={self.is_training: False,
                                           self.validation_loss: val_loss})
        sv.summary_writer.add_summary(val_sum, epoch_num)
        print("Epoch [{}] Validation Loss: {}".format(
            epoch_num, val_loss))
        # Model Saving
        if val_loss < self.min_val_loss:
            self.save(sess, self.config['log_dir'], 'best')
            self.min_val_loss = val_loss
        if epoch_num % self.config['save_freq'] == 0:
            self.save(sess, self.config['log_dir'], epoch_num)
        # Switch between training of coarse and fine network
        if epoch_num >= self.config['switching_epoch'] and self.do_fine_tuning == False:
            self.save(sess, self.config['log_dir'], 'switch')
            self.do_fine_tuning = True
            print("------------------------")
            print("Switched: Coarse to Fine")
            print("------------------------")

    def build_test_graph(self):
        """This graph will be used for testing. In particular, it will
           compute the loss on a testing set, or some other utilities.
        """
        self.input_rgb = tf.placeholder(tf.uint8, [1, None, None, 3],
                                        name='input_rgb')
        self.depth_labels = tf.placeholder(tf.uint8, [1, None, None, 1],
                                        name='depth_labels')
        input_shape = tf.shape(self.input_rgb)[1:3]
        rgb_image = self.preprocess_rgb(tf.squeeze(self.input_rgb, 0),
                                        is_training=tf.constant(False))
        depth_labels = self.preprocess_depth(tf.squeeze(self.depth_labels, 0),
                                        is_training=tf.constant(False))
        rgb_image = tf.expand_dims(rgb_image, 0)
        depth_labels = tf.expand_dims(depth_labels, 0)

        with tf.name_scope("coarse_prediction"):
            coarse_depth = eigen_coarse(rgb_image, is_training=False)

        with tf.name_scope("fine_prediction"):
            # predict fine depth map from coarse depth and RGB input image
            depth_predictions = eigen_fine(rgb_image, coarse_depth, is_training=False)

        with tf.name_scope("test_loss"):
            test_loss = self.loss_mse_scale_invariant(depth_predictions, depth_labels,
                                                 alpha=0.5)

        # TODO handle bitdepth
        # Scale and convert depth prediction to match imput
        test_prediction = tf.image.resize_bilinear(depth_predictions, input_shape)
        self.test_prediction = tf.cast(tf.clip_by_value(test_prediction*255., 0, 255), tf.uint8)
        self.test_loss = test_loss

class LearnerResnet(Learner):
    def __init__(self):
        Learner.__init__(self)

    def build_train_graph(self):
        is_training_ph = tf.placeholder(tf.bool, shape=(), name="is_training")

        with tf.name_scope("data_loading"):
            print('Parsing training directory...')
            train_batch, n_samples_train = self.generate_batches(
                    self.config['train_dir'], is_training=tf.constant(True))
            print('Parsing validation directory...')
            val_batch, n_samples_test = self.generate_batches(
                    self.config['val_dir'], is_training=tf.constant(False))
            current_batch = tf.cond(is_training_ph, lambda: train_batch,
                                                    lambda: val_batch)
            image_batch, depth_labels = current_batch[0], current_batch[1]

        with tf.name_scope("predict"):
            # predict fine depth map from coarse depth and RGB input image
            depth = resnet50(image_batch, is_training=True,
                             l2_reg_scale=self.config['l2_reg_scale'])

        with tf.name_scope("loss"):
            loss_mse = self.loss_mse_scale_invariant(depth, depth_labels, alpha=0.5)
            loss_l1 = tf.reduce_mean(tf.abs(depth - depth_labels))

        with tf.name_scope("train_op"):
            reg_losses = tf.reduce_sum(
                          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            train_vars = [var for var in tf.trainable_variables()]
            optimizer  = tf.train.AdamOptimizer(
                    self.config['learning_rate'], self.config['beta1'])
            self.grads_and_vars = optimizer.compute_gradients(
                    loss_l1+reg_losses, var_list=train_vars)
            #Select train operation depening on train mode coarse/fine
            self.train_op = optimizer.apply_gradients(self.grads_and_vars)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.incr_global_step = tf.assign(self.global_step, self.global_step+1)

        self.depth_prediction = depth
        self.depth_labels = depth_labels
        self.input_img = image_batch
        self.loss_mse = loss_mse
        self.loss_l1 = loss_l1
        self.train_steps_per_epoch = int(math.ceil(n_samples_train
                                                  /self.config['batch_size']))
        self.val_steps_per_epoch = int(math.ceil(n_samples_test
                                                /self.config['batch_size']))
        self.is_training = is_training_ph

    def collect_summaries(self):
        """Collects all summaries to be shown in the tensorboard"""
        tf.summary.scalar("Train Loss",
                self.loss_mse, collections=["step_sum"])
        tf.summary.scalar("Train Loss L1",
                self.loss_l1, collections=["step_sum"])
        tf.summary.image("Train Depth Prediction", self.depth_prediction,
                collections=["step_sum"], max_outputs=3)
        tf.summary.image("Train GT Depth Image", self.depth_labels,
                collections=["step_sum"], max_outputs=3)
        tf.summary.image("Train Input RGB Image", self.input_img,
                collections=["step_sum"], max_outputs=3)
        self.step_sum = tf.summary.merge(tf.get_collection("step_sum"))

        tf.summary.image("Validation Depth Prediction", self.depth_prediction,
                collections=["val_sum"], max_outputs=3)
        tf.summary.image("Validation GT Depth Image", self.depth_labels,
                collections=["val_sum"], max_outputs=3)
        tf.summary.image("Validation Input RGB Image", self.input_img,
                collections=["val_sum"], max_outputs=3)
        self.validation_loss = tf.placeholder(tf.float32, [])
        tf.summary.scalar("Validation_Loss", self.validation_loss,
                          collections = ["val_sum"])
        self.val_sum = tf.summary.merge(tf.get_collection("val_sum"))

    def train(self, config):
        """High level train function.
        Args:
            config: Configuration dictionary
        Returns:
            None
        """
        self.config = config
        self.build_train_graph()
        self.collect_summaries()
        self.min_val_loss = float('inf') # Initialize to max value

        parameter_count_backbone = tf.reduce_sum(
                [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables(
                    scope="resnet")])
        parameter_count_head = tf.reduce_sum(
                [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables(
                    scope="head")])
        self.saver = tf.train.Saver([var for var in \
                tf.trainable_variables()] +  [self.global_step], max_to_keep=3)

        # Find restoreable variables from pre-trained resnet backbome
        reader = tf.train.NewCheckpointReader(self.config['init_weights'])
        checkpoint_shapes = reader.get_variable_to_shape_map()
        checkpoint_names = set(checkpoint_shapes.keys())
        model_names = set([v.name.split(':')[0] for v in tf.trainable_variables()])
        found_names = model_names & checkpoint_names
        found_variables = []
        with tf.variable_scope('', reuse=True):
            for name in found_names:
                var = tf.get_variable(name)
                found_variables.append(var)
        self.saver_resnet = tf.train.Saver(var_list=found_variables)

        sv = tf.train.Supervisor(logdir=os.path.join(config['log_dir'],config['name']),
                                 save_summaries_secs=0,
                                 saver=None)
        with sv.managed_session() as sess:
            print("Number of trainable params backbone: {}".format(
                sess.run(parameter_count_backbone)))
            print("Number of trainable params head: {}".format(
                sess.run(parameter_count_head)))

            # Restore resnet pre-trained weights
            self.saver_resnet.restore(sess, self.config['init_weights'])
            print("Successfully restored %d pre-trained weights" % len(found_names))
            print("-------------------------------------------------")

            # Restore training checkpoint
            if config['resume_train']:
                print("Resume training from previous checkpoint")
                checkpoint = tf.train.latest_checkpoint(config['resume_weights'])
                self.saver.restore(sess, checkpoint)

            progbar = Progbar(target=self.train_steps_per_epoch)

            # training loop
            for step in count(start=1):
                if sv.should_stop():
                    break
                start_time = time.time()
                # Variables that are fetched every step
                fetches = { "train" : self.train_op,
                            "global_step" : self.global_step,
                            "incr_global_step": self.incr_global_step}
                # Variables that are only fetched when summary is saved
                if step % config['summary_freq'] == 0:
                    fetches["loss"] = self.loss_mse
                    fetches["summary"] = self.step_sum

                # Runs the series of operations
                results = sess.run(fetches,
                                   feed_dict={self.is_training: True})

                gs = results["global_step"]
                if step % config['summary_freq'] == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)

                if step % self.train_steps_per_epoch == 0:
                    # This differ from the last when resuming training
                    progbar.update(self.train_steps_per_epoch)
                    train_epoch = int(gs / self.train_steps_per_epoch)
                    self.epoch_end_callback(sess, sv, train_epoch, config)
                    progbar = Progbar(target=self.train_steps_per_epoch)
                    if (train_epoch == self.config['max_epochs']):
                        print("-------------------------------")
                        print("Training completed successfully")
                        print("-------------------------------")
                        break
                else:
                    progbar.update(step % self.train_steps_per_epoch)

    def epoch_end_callback(self, sess, sv, epoch_num, config):
        # Check epoch number to switch from coarse to fine network
        fetches = { "loss" : self.loss_mse }

        val_loss = 0
        for i in range(self.val_steps_per_epoch):
            results = sess.run(fetches, feed_dict={self.is_training: False})
            val_loss += results['loss']
        val_loss = val_loss / self.val_steps_per_epoch
        val_sum = sess.run(self.val_sum, feed_dict ={
                                                self.is_training: False,
                                                self.validation_loss: val_loss})
        sv.summary_writer.add_summary(val_sum, epoch_num)
        print("Epoch [{}] Validation Loss: {}".format(
            epoch_num, val_loss))
        # Model Saving
        if val_loss < self.min_val_loss:
            self.save(sess, self.config['log_dir'], 'best')
            self.min_val_loss = val_loss
        if epoch_num % self.config['save_freq'] == 0:
            self.save(sess, self.config['log_dir'], epoch_num)

    def build_test_graph(self):
        """This graph will be used for testing. In particular, it will
           compute the loss on a testing set, or some other utilities.
        """
        self.input_rgb = tf.placeholder(tf.uint8, [1, None, None, 3],
                                        name='input_rgb')
        self.depth_labels = tf.placeholder(tf.uint8, [1, None, None, 1],
                                        name='depth_labels')
        input_shape = tf.shape(self.input_rgb)[1:3]

        rgb = self.preprocess_rgb(tf.squeeze(self.input_rgb, 0),
                                        is_training=tf.constant(False))
        depth_labels = self.preprocess_depth(tf.squeeze(self.depth_labels, 0),
                                        is_training=tf.constant(False))

        rgb = tf.expand_dims(rgb, 0)
        depth_labels = tf.expand_dims(depth_labels, 0)

        with tf.name_scope("test_op"):
            depth = resnet50(rgb, is_training=True,
                             l2_reg_scale=self.config['l2_reg_scale'])

        with tf.name_scope("test_loss"):
            test_loss = self.loss_mse_scale_invariant(depth, depth_labels, alpha=0.5)

        # TODO handle bitdepth
        # Scale and convert depth prediction to match imput
        test_prediction = tf.image.resize_bilinear(depth, input_shape)
        self.test_prediction = tf.cast(tf.clip_by_value(test_prediction*255., 0, 255), tf.uint8)
        self.test_loss = test_loss
