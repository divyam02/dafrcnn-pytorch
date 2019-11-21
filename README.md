# Domain Adaptive Faster R-CNN

A PyTorch implementation of the CVPR 2018 paper [Domain Adaptive Faster R-CNN for Object Detection in the Wild](https://arxiv.org/pdf/1803.03243).
The original code used by the authors can be found [here](https://github.com/yuhuayc/da-faster-rcnn).

This implementation is built on [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch). You may find it to be helpful for debugging.

## Usage
Ensure all prerequisites mentioned [here](https://github.com/jwyang/faster-rcnn.pytorch) are satisfied by your machine. This model is meant for `pytorch 0.4.0`.

Install python dependencies with `pip install -r requirements.txt`.

Ensure all images used for training (source) and testing have annotations (in Pascal VOC format).

Refer to dummy formatting in `./lib/datasets/pascal_voc.py` files to make your dataset usable.

Changes will be made in the future for ease of use.

### Compile
Compile cuda dependencies with `cd lib && sh make.sh`. Incase cuda is not on `$PATH` use `make2.sh`.

### Train
Add the source and target datasets (in Pascal VOC format) in the `src/` and `tar/` folders respectively.

Modify `factory.py` as given in the dummy formatting so it is usable.

Run training as:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python new_trainval.py \
                   --dataset $DATA SET NAME\
                   --adaption_lr True --net res101 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda
```
### Test
For testing your model, ensure the model is placed in the target dataset folder. Run:
```
python test_net.py --dataset $TARGET_DATASET_NAME --net res101 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda --load_dir $path/to/dir
```
Specify the specific model session, checkepoch and checkpoint, e.g., SESSION=1, EPOCH=5, CHECKPOINT=5931.

### Demo
Follow instructions given [here](https://github.com/jwyang/faster-rcnn.pytorch#demo)

## Benchmarks
________________________________________________________________________________________________________________________

Source dataset  | Target dataset  |
-----------------|-----------------|
Cityscapes  | Foggy Cityscapes

Model | Mean AP | Person AP | Rider AP  | Car AP  | Truck AP  | Bus AP  | Train AP  | Motorcycle AP | Bicycle AP  |
------|---------|-----------|-----------|---------|-----------|---------|-----------|---------------|-------------|
Faster R-CNN | 18.8  | 17.8  | 23.6  | 27.1  | 11.9  | 23.8  | 9.1 | 14.4  | 22.8  |
Y.Chen et al  | 27.6  | 25.0  | 31.0  | 40.5  | 22.1  | 35.3  | 20.2  | 20.0  | 27.1  |
dafrcnn-pytorch | 28.6  | 25.7  | 40.1  | 36.0  | 20.1  | 33.9  | 17.3  | 24.3  | 31.3  |

________________________________________________________________________________________________________________________

Source dataset  | Target dataset  |
----------------|-----------------|
Zoo_Leipzig_Individuen (Chimpanzees)  | (Name to be updated) (Monkeys)

Model | Mean AP (Face)
------|---------------|
dafrcnn-pytorch | 89.1  |
Faster RCNN | 88.2  |

________________________________________________________________________________________________________________________

Source dataset  | Target dataset  |
----------------|-----------------|
KITTI | Cityscapes

Model | Mean  AP  (Car)
------|------------|
dafrcnn-pytorch | 39.7  |
Faster RCNN (VGG16) | 35.3  |

________________________________________________________________________________________________________________________

This particular experiment was with temporal data and is not publically available. Uses **camera trap images**.
Consists of images from Chitals, Tigers, Sambar and Leopards.

Source dataset  | Target dataset  |
----------------|-----------------|
Species_2014  | Species_2018  |

Model | Mean AP | tiger AP  | sambar AP | leopard AP  | chital AP |
------|------|-----------|------------|------------|--------|
dafrcnn-pytorch | 83.7  | 79  | 83.1  | 90.7  | 82.2 |
Faster RCNN | 81.6  | 77.9  | 83.3  | 80.4  | 81.6  |

________________________________________________________________________________________________________________________

Comparing results for [this paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Inoue_Cross-Domain_Weakly-Supervised_Object_CVPR_2018_paper.pdf)

Source dataset  | Target dataset  |
----------------|-----------------|
Pascal VOC 2007 + 2012  | Clipart1K | 

Model | Mean AP |
-------|--------|
dafrcnn-pytorch | 30.6  |
Faster RCNN | 28.3  |

________________________________________________________________________________________________________________________

Source dataset  | Target dataset  |
----------------|-----------------|
SIM10K  | Cityscapes  |
dafrcnn-pytorch | 34.3  |
Faster RCNN | 30.6  |


## Current Work
Working on a under-segmentation problem observed in our target data set. Example image:
![how many tigers?](https://github.com/divyam02/dafrcnn-pytorch/blob/master/resources/in_bala_01_c027a_05052018073503_P1166.jpg)

## Acknowledgements
This code is built on the python implementation of the Faster-RCNN [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)

Completed as an independent study under Dr. Saket Anand
