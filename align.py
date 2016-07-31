# coding=utf-8
'''
功能：从人脸框中检测眼睛和嘴巴的中心点
参数：argv[1] : 图片存放的目录
      argv[2] : 人脸框.rect文件存放的目录
      argv[3] : landmark-3文件想要保存的目录
      argv[4] : 对准后的人脸图片要储存的目录
'''

import sys
import os
import openface
import numpy as np
import dlib
import cv2

def align(image_dir, boundingbox_dir, landmark_dir, rotated_image_dir):
    
    if not os.path.exists(rotated_image_dir):os.mkdir(rotated_image_dir)
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, 'shape_predictor_68_face_landmarks.dat')
    
    face_detector = openface.AlignDlib(model_path)
    
    image_list = [f for f in os.listdir(input_dir) if (f.endswith('.png') or f.endswith('.jpg'))]
    
    for filename in image_list:
        
        image_path = os.path.join(input_dir, filename)
        boundingbox_path =  os.path.join(boundingbox_dir, filename+'.rect')
        rotated_image_path =  os.path.join(rotated_image_dir, filename)
        if os.path.exists(boundingbox_path):
            face_rect_np = np.fromfile(boundingbox_path, dtype=np.int32)
            x = long(face_rect_np[0])
            y = long(face_rect_np[1])
            w = long(face_rect_np[2])
            h = long(face_rect_np[3])
            bb = dlib.rectangle(x,y,x+w,y+h)
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            alignedFace = face_detector.align(96, image, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            
            alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_RGB2BGR)
            cv2.imwrite(rotated_image_path, alignedFace)
 
    
if __name__ == '__main__':
    print('*'*79)
    print('Dlib face alignment')

    assert len(sys.argv) == 5, "!!! 4 arguments are needed !!!"
    input_dir = os.path.abspath(sys.argv[1])
    boundingbox_dir = os.path.abspath(sys.argv[2])
    landmark_dir = os.path.abspath(sys.argv[3])
    rotated_image_dir = os.path.abspath(sys.argv[4])

    align(input_dir, boundingbox_dir, landmark_dir, rotated_image_dir)

    
