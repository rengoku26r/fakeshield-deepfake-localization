import argparse
import cv2
import sys
import numpy as np
import datetime
import os
import glob
import time
from retinaface import RetinaFace
from os.path import join
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--raw', help='input data folder')
parser.add_argument('--box', help='bounding box file')
opt = parser.parse_args()

ifolder = opt.raw
box_output = open(opt.box + 'box.txt', 'w')
box_output_pickle = open(opt.box + 'box.pickle', 'wb')

thresh = 0.8
count = 1
gpuid = 0
detector = RetinaFace('./model/R50', 0, gpuid, 'net3')

files = sorted(os.listdir(ifolder))
count_current = 0
for f in files:
  if(count_current % 1000 == 0):
    print(str(count_current) + ":" + time.asctime( time.localtime(time.time()) ))
  scales = [512, 990]
  img = cv2.imread(os.path.join(ifolder,f))
  im_shape = img.shape
  target_size = scales[0]
  max_size = scales[1]
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])
  im_scale = float(target_size) / float(im_size_min)
  if np.round(im_scale * im_size_max) > max_size:
      im_scale = float(max_size) / float(im_size_max)

  scales = [im_scale]
  flip = False

  faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
  print('{:d}/{:d}'.format(count_current, len(files)), end='\r')
  count_current += 1
  if faces is None or faces.shape[0] == 0:
    print("processing failed: " + f)
    continue
  wh_largest = 0
  index_largest = 0
  i = 0
  for face in faces:
    wh = face[2] - face[0] + face[3] - face[1]
    if wh > wh_largest:
      wh_largest = wh
      index_largest = i
    i += 1

  box_output.write(f + ": " + str(faces) + "\n" + str(landmarks) + "\n")
  pickle.dump([f, faces, landmarks], box_output_pickle)

print('{:d}/{:d}'.format(count_current, len(files)))
box_output.close()
box_output_pickle.close()
