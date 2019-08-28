#!/usr/bin/env python
import cv2
import rospy
import tensorflow as tf
import numpy as np


from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from depth_learner.models.base_learner import LearnerResnet, LearnerEigen

bridge = CvBridge()

class Network(object):
    def __init__(self, config):

        self.config = config

        self.pub = rospy.Publisher('/depth_images_pred', Image, queue_size=1)

        if self.config['model_name'] == 'resnet':
            self.learner = LearnerResnet()
        elif self.config['model_name'] == 'eigen':
            self.learner = LearnerEigen()

        self.learner.setup_inference(config)

        self.saver = tf.train.Saver([var for var in tf.trainable_variables()])

    def load_model(self, sess):
        self.sess = sess
        checkpoint = self.config['inference_model'] + '/model-40'
        try:
            self.saver.restore(self.sess, checkpoint)
            print("Restored checkpoint file {}".format(checkpoint))
            print("--------------------------------------------------")
        except:
            raise ValueError("Could not restore model checkpoint.")

    def run(self, sess):
        self.sess = sess
        while not rospy.is_shutdown():
            rgb_image = None

            # Reading image and processing it through the network
            try:
                rgb_image = rospy.wait_for_message("/rgb_images", Image)
            except:
                print("CNN inference: Could not aquire image data")
                break
            try:
                cv_rgb_image = bridge.imgmsg_to_cv2(rgb_image)
            except CvBridgeError as e:
                print(e)
                continue

            inputs = {}
            inputs['rgb'] = cv_rgb_image[None]
            start = rospy.get_rostime()
            results = self.learner.inference(inputs, sess)
            end = rospy.get_rostime()
            inference_time = end-start
            rospy.loginfo("Infered depth: runtime %d ms", inference_time.to_nsec()*1e-6)
            depth_prediction = np.clip(np.squeeze(results['depth_prediction']), 0, 255)
            imgmsg_depth = bridge.cv2_to_imgmsg(depth_prediction, encoding="mono8")
            imgmsg_depth.header.stamp = rospy.Time.now()
            imgmsg_depth.header.frame_id = "/world"

            self.pub.publish(imgmsg_depth)
