import os
with open('debug_caltech.txt', 'r') as f:
	bad_files = f.readlines()
	for bad_file in bad_files:
		os.rename("/home/divyam/dafrcnn/dafrcnn-pytorch/data/src/caltech/coco/images/"+"COCO_train2014_"+bad_file[:-1]+".jpg", 
					"/home/divyam/dafrcnn/dafrcnn-pytorch/data/src/caltech/coco/images_no_annots/"+"COCO_train2014_"+bad_file[:-1]+".jpg")