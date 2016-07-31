import align_face
import cv2
import numpy as np
from get_feature import get_feature
import logging
    
aligner = align_face.AlignDlib('shape_predictor_68_face_landmarks.dat')

def align(filename):

    bgrImg = cv2.imread(filename, 1)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    logging.info('detect')
    bb = aligner.getLargestFaceBoundingBox(rgbImg)
    logging.info('landmarks')
    landmarks = aligner.findLandmarks(rgbImg, bb)
    logging.info('align')
    alignedFace = aligner.align(128, rgbImg, landmarks)
    logging.info('feature')
    alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_RGB2GRAY)
    feature = get_feature(alignedFace)
    return feature
    
def compare(feature1, feature2):
    f1 = np.matrix(feature1)    
    f2 = np.matrix(feature2)
    
    similarity = (f1 * f2.transpose() / np.linalg.norm(f1) / np.linalg.norm(f2))[0,0]
    
    if similarity < 0 : similarity = 0
    
    return similarity
