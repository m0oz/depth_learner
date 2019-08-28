import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import slim

from .backbones import resnet_v2 as resnet

def eigen_coarse(img_input, is_training, l2_reg_scale=1e-5, scope='coarse_prediction'):
    """
    Reference implementation of CNN for single image depth prediction
    as proposed by Eigen et al
    # Arguments
        img_input: batch of RGB input images
    # Returns
        output: pixel depth map of the same dimension as input image
        where high intensity represents close areas and
        low intensity represents far areas
    """
    with tf.variable_scope(scope):
        coarse1_conv = layers.convolution2d(img_input, num_outputs=96, kernel_size=11,
            padding='VALID', stride=4,
            weights_regularizer=layers.l2_regularizer(l2_reg_scale),
            activation_fn=tf.nn.relu)
        coarse1 = layers.max_pool2d(coarse1_conv, kernel_size=3,
            padding='VALID', stride=2)
        coarse2_conv = layers.convolution2d(coarse1, num_outputs=256, kernel_size=5,
            padding='VALID', stride=1,
            weights_regularizer=layers.l2_regularizer(l2_reg_scale),
            activation_fn=tf.nn.relu)
        coarse2 = layers.max_pool2d(coarse2_conv, kernel_size=3,
            padding='VALID', stride=2)
        coarse3_conv = layers.convolution2d(coarse2, num_outputs=384, kernel_size=3,
            padding='VALID', stride=1,
            weights_regularizer=layers.l2_regularizer(l2_reg_scale),
            activation_fn=tf.nn.relu)
        coarse4_conv = layers.convolution2d(coarse3_conv, num_outputs=384, kernel_size=3,
            padding='VALID', stride=1,
            weights_regularizer=layers.l2_regularizer(l2_reg_scale),
            activation_fn=tf.nn.relu)
        coarse5_conv = layers.convolution2d(coarse4_conv, num_outputs=256, kernel_size=3,
            padding='VALID', stride=1,
            weights_regularizer=layers.l2_regularizer(l2_reg_scale),
            activation_fn=tf.nn.relu)
        coarse5 = layers.flatten(coarse5_conv)
        coarse6 = layers.fully_connected(coarse5, num_outputs=4096,
            weights_regularizer=layers.l2_regularizer(l2_reg_scale))
        coarse6_dropout = layers.dropout(coarse6, keep_prob=0.8,
            is_training=is_training)
        coarse7 = layers.fully_connected(coarse6_dropout, num_outputs=4070,
            weights_regularizer=layers.l2_regularizer(l2_reg_scale),
            activation_fn=None)
        coarse7 = tf.reshape(coarse7,[-1,55,74,1])

        output = coarse7
    return output

def eigen_fine(img_input, depth_coarse, is_training, l2_reg_scale=1e-5, scope='fine_prediction'):
    """
    Reference implementation of CNN for single image depth prediction
    as proposed by Eigen et al
    # Arguments
        img_input: batch of RGB input images
        coarse_depth: output depth image from coarse net
    # Returns
        output: pixel depth map of the same dimension as input image
        where high intensity represents close areas and
        low intensity represents far areas
    """
    with tf.variable_scope(scope):
        fine1_conv = layers.convolution2d(img_input, num_outputs=63, kernel_size=9,
            padding='VALID', stride=2,
            weights_regularizer=layers.l2_regularizer(l2_reg_scale),
            activation_fn=tf.nn.relu)
        fine1 = layers.max_pool2d(fine1_conv, kernel_size=3,
            padding='SAME', stride=2)
        fine2 = tf.concat([fine1, depth_coarse], 3)
        fine3_conv = layers.convolution2d(fine2, num_outputs=64, kernel_size=5,
            padding='SAME', stride=1,
            weights_regularizer=layers.l2_regularizer(l2_reg_scale),
            activation_fn=tf.nn.relu)
        fine4_conv = layers.convolution2d(fine3_conv, num_outputs=1, kernel_size=5,
            padding='SAME', stride=1,
            weights_regularizer=layers.l2_regularizer(l2_reg_scale),
            activation_fn=None)

        output = fine4_conv
    return output

def resnet50(img_input, is_training, l2_reg_scale):
    with tf.name_scope('resnet'):
        input_shape = img_input.get_shape().as_list()
        with slim.arg_scope(resnet.resnet_arg_scope()):
            _, encoder = resnet.resnet_v2_50(
                                        img_input,
                                        is_training=is_training,
                                        global_pool=False,
                                        scope='resnet_v2_50')
        features = encoder['resnet_v2_50/block4']
    with tf.variable_scope('head'):
        x = slim.conv2d(features, 1024 , 1, normalizer_fn=slim.batch_norm)
        x = slim.conv2d_transpose(x, 512, kernel_size=3, stride=2,
                                  weights_regularizer=slim.l2_regularizer(l2_reg_scale))
        x = slim.conv2d_transpose(x, 256, kernel_size=3, stride=2,
                                  weights_regularizer=slim.l2_regularizer(l2_reg_scale))
        x = slim.conv2d_transpose(x, 128, kernel_size=4, stride=2,
                                  weights_regularizer=slim.l2_regularizer(l2_reg_scale))
        x = slim.conv2d_transpose(x, 64, kernel_size=4, stride=2,
                                  weights_regularizer=slim.l2_regularizer(l2_reg_scale))
        x = slim.convolution2d(x, 1, 3, activation_fn=None)
        x = tf.image.resize_bilinear(x, (228, 304))
    return x
