import gflags
FLAGS = gflags.FLAGS

# Required flags
gflags.DEFINE_string('config', './configs/resnet.yaml', 'Specify path to config file')
gflags.DEFINE_string('name', 'test', 'Experiment name')

# Train parameters
gflags.DEFINE_bool('resume_train', False, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_integer('batch_size', 16, 'Batch size in training and evaluation')

# GPU threading parameters
gflags.DEFINE_integer('num_threads', 6, 'Number of threads reading and '
                      '(optionally) preprocessing input files into queues')
gflags.DEFINE_integer('capacity_queue', 20, 'Capacity of input queue. A high '
                      'number speeds up computation but requires more RAM')

# Log parameters
gflags.DEFINE_integer("summary_freq", 200, "Logging every log_freq iterations")
gflags.DEFINE_integer("save_freq", 1,
                      "Save the latest model every save_freq epochs")
