import os
with open('test.txt', 'w') as f:
	for file in os.listdir():
		if file=='to_coco_format.py' or file=='create_test_file.py':
			continue
		else:
			f.write(file+"\n")