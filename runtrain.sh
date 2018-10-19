#!/bin/bash
while [[kill -0 $0]]
do
   echo "$PID is running"
   # Do something knowing the pid exists, i.e. the process with $PID is running
done

python trainval_net.py --da True --src city --tar fcity --net res101 --bs 1 --lr 0.001 --save_dir data/pretrained_model --cuda --lr_decay_step 50000 2>&1 | tee debug/train_loop_04.09.18_1
echo "Executed"
