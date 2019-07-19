import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np

obj_struct = {}
with open('check.txt', 'r') as f:
  for filename in f.readlines():
    try:
      filename = filename[:-1]+'.xml'
      tree = ET.parse(filename)
      for obj in tree.findall('object'):
        name = obj.find('name').text
        obj_struct[str(name)] = True
    except:
      print("weird file", filename)
  print(obj_struct.keys())