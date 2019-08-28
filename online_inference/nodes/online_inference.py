#!/usr/bin/env python

import rospy
import re
from Network import Network
import os, datetime
import tensorflow as tf
import sys
import gflags
import yaml

def run_network(config):

    rospy.init_node('online_inference', anonymous=True)

    # RUN NETWORK
    with tf.Session() as sess:
        network = Network.Network(config)
        network.load_model(sess)
        network.run(sess)

def loadParameters():
    param_names = rospy.get_param_names()
    config = {}
    for key in param_names:
        value = rospy.get_param(key)
        _, key = key.rsplit('/', 1)
        config.update({key: value})
    return config

if __name__ == "__main__":
    config = loadParameters()
    run_network(config)
