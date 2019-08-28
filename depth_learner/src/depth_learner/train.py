import tensorflow as tf
import pprint
import random
import numpy as np
from models.base_learner import LearnerResnet, LearnerEigen
import os
import gflags
import sys
import yaml
from shutil import copyfile

from common_flags import FLAGS

#####################################
# THIS FILE SHOULD REMAIN UNCHANGED #
#####################################

def _main():
    # Set random seed for training
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    flags_dict = {}
    for key in FLAGS.__flags.keys():
        flags_dict[key] = getattr(FLAGS, key)

    with open(FLAGS.config, 'r') as f:
        yaml_dict = yaml.load(f)

    config = {}
    config.update(flags_dict)
    config.update(yaml_dict)
    pp = pprint.PrettyPrinter()
    pp.pprint(config)

    if not os.path.exists(os.path.join(config['log_dir'], config['name'])):
        os.makedirs(os.path.join(config['log_dir'], config['name']))
    # Save config yaml to log directorys
    copyfile(config['config'],
             os.path.join(config['log_dir'], config['name'], 'config.yaml'))

    if config['model_name'] == 'resnet':
        trl = LearnerResnet()
    if config['model_name'] == 'eigen':
        trl = LearnerEigen()
    trl.train(config)

def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    _main()

if __name__ == "__main__":
    main(sys.argv)
