import os
import shutil
import cv2
path1 = r'1/label'
path2 = r'VOCdevkit/VOC2007/SegmentationClass'

for file in os.listdir(path1):
    file_path = os.path.join(path1, file)
    img = cv2.imread(file_path, 0) /255
    cv2.imwrite(os.path.join(path2, file.replace('.tif', '.png')), img)   