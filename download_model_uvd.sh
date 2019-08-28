mkdir -p depth_learner/src/depth_learner/experiments/resnet50_0306
cd depth_learner/src/depth_learner/experiments/resnet50_0306
echo "Downloading and extracting pretrained model for uvd dataset..."
tar x < <(wget -q -O - https://polybox.ethz.ch/index.php/s/hAnajtcxHnenV34/download)
echo "Complete"
echo "-------------------------"
cd ../../../../..
