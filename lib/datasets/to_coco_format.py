import os
for img_name in os.listdir():
	if img_name == "to_coco_format.py":
		continue
	os.rename(img_name, "COCO_train2014_"+img_name)