# NOTE
## Please know that this branch has not been updated yet. All users are requested to use the [cleaning branch](https://github.com/divyam02/dafrcnn-pytorch/tree/cleaning) of the repository.

# Domain Adaptive Faster R-CNN

A PyTorch implementation of the CVPR 2018 paper [Domain Adaptive Faster R-CNN for Object Detection in the Wild.](https://arxiv.org/pdf/1803.03243)
The original code used by the authors can be found [here.](https://github.com/yuhuayc/da-faster-rcnn)

## Usage
Ensure all rerequisites mentioned [here](https://github.com/jwyang/faster-rcnn.pytorch) are satisfied by your machine.
Ensure all images used for training (source) and testing have annotations (in Pascal VOC format).
Modify [this](https://github.com/divyam02/dafrcnn-pytorch/blob/e22be667140b7dcfca4325a8e7bb9f325048e124/trainval_net_x.py#L696) line in `trainval_net_x.py` to the directory of the target dataset.

Changes will be made in the future for ease of use.

### Train
Add the source and target datasets (in Pascal VOC format) in the `src/` and `tar/` folders respectively.
Modify `factory.py` so that your dataset is usable.
Run training as:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net_x.py \
                   --src $SOURCE_DATASET_NAME --tar $TARGET_DATASET_NAME \
                   --da True --adaption_lr True --net res101 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda
```
### Test
If you want to evaluate the detection performance of a pre-trained model on your testset run:
```
python test_net_x.py --dataset $TARGET_DATASET_NAME --net res101 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
```
Specify the specific model session, checkepoch and checkpoint, e.g., SESSION=1, EPOCH=5, CHECKPOINT=5931.

### Demo
Follow instructions given [here](https://github.com/jwyang/faster-rcnn.pytorch#demo)

## Acknowledgements
This code is built on the python implementation of the Faster-RCNN [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)

Part of an ongoing project under Dr. Saket Anand
