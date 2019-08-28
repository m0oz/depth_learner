import numpy as np
import re
import os

class DirectoryIterator():
    """
    Generic class to parse folder structure and build file list

    # Arguments
       directory: Path to the root directory to read data from.
       follow_links: Bool, whether to follow symbolic links or not

    """

    def __init__(self, directory, dataset, follow_links=False):

        self.directory = directory
        self.follow_links = follow_links
        self.filenames = []
        self.samples = 0
        self.experiments = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                self.experiments.append(subdir)

        self.num_experiments = len(self.experiments)
        self.formats = {'png'}
        print("----------------------------------------------------")
        print("Loading the following formats {}".format(self.formats))
        print("----------------------------------------------------")

        for subdir in self.experiments:
            subpath = os.path.join(directory, subdir)
            try:
                # Call function to decode this specific dataset
                if dataset == 'nyu':
                    self._decode_experiment_dir_NYU(subpath)
                elif dataset == 'scenenet':
                    self._decode_experiment_dir_SceneNet(subpath)
                elif dataset == 'sunrgbd':
                    self._decode_experiment_dir_SUNRGBD(subpath)
                elif dataset == 'unreal':
                    self._decode_experiment_dir_Unreal(subpath)
                elif dataset == 'uvd':
                    self._decode_experiment_dir_UVD(subpath)
            except:
                raise ImportWarning("Image reading in {} failed".format(
                                        subpath))
        if self.samples == 0:
            raise IOError("Did not find any file in the dataset folder")

        print('Length list %d' % len(self.filenames))
        print('Found {} images belonging to {} experiments:'.format(
                self.samples, self.num_experiments))
        print(self.experiments)

    def _recursive_list(self, subpath):
        return os.walk(subpath, followlinks=self.follow_links)

    def _decode_experiment_dir_NYU(self, dir_subpath):
        """
        function to decode images and depths for the NYU dataset
        We assume that the folder structure is:
        root_folder/
               Images1/
                    00001.jpg (rgb)
                    00001.png (depth)
                    00002.jpg (rgb)
                    00002.png (depth)
                    ...
               Images2/
                    00001.jpg (rgb)
                    00002.png (depth)
                    ...

        # Arguments
           directory: Path to the root directory to read data from.
        """
        image_dir_path = dir_subpath
        for root, _, files in self._recursive_list(image_dir_path):
            sorted_files = sorted(files,
                    key = lambda fname: int(re.search(r'\d+',fname).group()))
            for frame_number, fname in enumerate(sorted_files):
                is_valid = False
                # Select all files that have one of the specified formats
                for extension in self.formats:
                    if fname.lower().endswith('depth.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    path_depth = os.path.join(root, fname)
                    # Store filenames for rgb images
                    # Replace .png by .jpg to select rgb images
                    [path_rgb, _] = path_depth.split('depth.png')
                    path_rgb = path_rgb + 'colors.png'
                    # Append both paths as tuple to filenames list
                    self.filenames.append([path_rgb, path_depth])
                    self.samples += 1

    def _decode_experiment_dir_SceneNet(self, dir_subpath):
        """
        Sub-class of DirectoryIterator
        Class for managing data loading of images and depths for the SceneNet dataset
        We assume that the folder structure is:
        root_folder/
               train_0/
                    0/
                        depth/
                            0.png
                            25.png
                            ...
                        photo/
                            0.jpg
                            25.jpg
               train_1/
                    0/
                        ...
               ...
               train_n/
                    ...
        # Arguments
           directory: Path to the root directory to read data from.
        """
        image_dir_path = dir_subpath
        # Append all experiment files to one filenames list
        i = 0

        for root, dirs, files in self._recursive_list(image_dir_path):
            if 'depth' in root:
                for fname_depth in files:
                    is_valid = False
                    for extension in self.formats:
                        if fname_depth.lower().endswith('.' + extension) == True:
                            is_valid = True
                    if is_valid:
                        path_depth = os.path.join(root, fname_depth)
                        [path, _] = re.split('depth', path_depth)
                        fname_rgb, _ = re.split('.png', fname_depth)
                        fname_rgb = fname_rgb + '.jpg'
                        path_rgb = os.path.join(path, 'photo/', fname_rgb)
                        self.filenames.append([path_rgb, path_depth])

                        self.samples += 1

    def _decode_experiment_dir_SUNRGBD(self, dir_subpath):
        """
        Sub-class of DirectoryIterator
        Class for managing data loading of images and depths for the SceneNet dataset
        We assume that the folder structure is:
        root_folder/
               kv1/
                    subexperiment0/
                        image0/
                            depth/
                                0.png
                            image/
                                0.jpg
                        image1/
                            depth/
                                ...
                    subexperiment1/
               kv2/
                    ...
        # Arguments
           directory: Path to the root directory to read data from.
        """
        image_dir_path = dir_subpath
        # Append all experiment files to one filenames list
        path_depth = ''
        path_rgb = ''
        for root, _, files in self._recursive_list(image_dir_path):
            if root.endswith('/depth'):
                for file in files:
                    path_depth = os.path.join(root, file)
            if root.endswith('/image'):
                for file in files:
                    path_rgb = os.path.join(root, file)

            # Check file extension
            is_valid = True
            for extension in self.formats:
                if path_depth.endswith('.' + extension) == False:
                    is_valid = False
            # Check if rgb and depth file ar
            root_depth = re.split('/depth/', path_depth)
            root_rgb = re.split('/image/', path_rgb)
            if root_depth[0] != root_rgb[0]:
                is_valid = False

            if is_valid:
                self.filenames.append([path_rgb, path_depth])
                self.samples += 1
                path_depth = ''
                path_rgb = ''

    def _decode_experiment_dir_Unreal(self, dir_subpath):
        """
        Sub-class of DirectoryIterator
        Class for managing data loading of images and depths for the SceneNet dataset
        We assume that the folder structure is:
        root_folder/
               00_D/
                    depth/
                        00000.png
                        00001.png
                        ...
                    rgb/
                        00000.png
                        00001.png
                        ...
               01_D/
                    ...
        # Arguments
           directory: Path to the root directory to read data from.
        """
        image_dir_path = dir_subpath
        # Append all experiment files to one filenames list
        path_depth = ''
        path_rgb = ''
        for root, _, files in self._recursive_list(image_dir_path):
            if root.endswith('/depth'):
                for file in files:
                    path_depth = os.path.join(root, file)

                    # Check file extension
                    [path, fname] = re.split('depth/', path_depth)
                    path_rgb = os.path.join(path, 'rgb', fname)
                    self.filenames.append([path_rgb, path_depth])
                    self.samples += 1

    def _decode_experiment_dir_UVD(self, dir_subpath):
        """
        Sub-class of DirectoryIterator
        Class for managing data loading of images and depths for the SceneNet dataset
        We assume that the folder structure is:
        root_folder/
               00/
                    DepthSR/
                        0001 02.png
                        0002 02.png
                        ...
                    Images/
                        0001 02.png
                        0002 02.png
                        ...
               01/
                    ...
        # Arguments
           directory: Path to the root directory to read data from.
        """
        image_dir_path = dir_subpath
        # Append all experiment files to one filenames list
        path_depth = ''
        path_rgb = ''
        for root, _, files in self._recursive_list(image_dir_path):
            if root.endswith('/DepthSR'):
                for file in files:
                    path_depth = os.path.join(root, file)

                    # Check file extension
                    [path, fname] = re.split('DepthSR/', path_depth)
                    path_rgb = os.path.join(path, 'Images', fname)
                    self.filenames.append([path_rgb, path_depth])
                    self.samples += 1
