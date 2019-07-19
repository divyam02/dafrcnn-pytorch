from PIL import Image
import os 
for img_file in os.listdir():
	try:
		print(img_file)
		img = Image.open(img_file)
		rgb_im = img.convert('RGB')
		#print(img_file[:-3])
		#assert 1<0
		rgb_im.save(img_file[:-3] + "jpg")
	except:
		print("failed:", img_file)