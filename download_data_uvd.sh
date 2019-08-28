mkdir -p depth_learner/src/depth_learner/data/weights/resnet_v2_50
cd depth_learner/src/depth_learner/data/weights/resnet_v2_50
echo "Downloading and extracting pre-trained resnet weights..."
tar xz < <(wget -q -O - http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)
echo "Complete"
echo "-------------------------"
cd ../../../../../..

mkdir -p depth_learner/src/depth_learner/data/uvd/val
mkdir -p depth_learner/src/depth_learner/data/uvd/test/09/Images
mkdir -p depth_learner/src/depth_learner/data/uvd/test/09/DepthSR
cd depth_learner/src/depth_learner/data/uvd/val
echo "Downloading and extracting UVD test split..."
tar xz < <(wget -q -O - http://www.sira.diei.unipg.it/supplementary/ral2016/datasets/uvd_test.tar.gz)
cd 09/Images
mv ???0* ???2* ???4* ???6* ???8* ../../../test/09/Images
cd ../DepthSR
mv ???0* ???2* ???4* ???6* ???8* ../../../test/09/DepthSR
echo "Complete"
echo "-------------------------"
cd ../../../../../../../..

mkdir -p depth_learner/src/depth_learner/data/uvd/train
cd depth_learner/src/depth_learner/data/uvd/train
echo "Downloading and extracting UVD train split... (This might take some minutes)"
tar xz < <(wget -q -O - http://www.sira.diei.unipg.it/supplementary/ral2016/datasets/uvd_train.tar.gz)
echo "Complete"
echo "-------------------------"
cd ../../../../../..
