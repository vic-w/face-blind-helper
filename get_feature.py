# coding=utf-8

import os
import sys
import cv2
import mxnet as mx
import numpy as np

from lightened_cnn import lightened_cnn_b_feature

ctx = mx.cpu()
model_prefix="lightened_cnn"
epoch=9999
neural_net = lightened_cnn_b_feature()
_, model_args, model_auxs = mx.model.load_checkpoint(model_prefix, epoch)

    
def get_feature(grayimage):
    img_arr = np.zeros((1, 1, 128, 128), dtype=float)
    img = np.expand_dims(grayimage, axis=0)
    img_arr[0][:] = img/255.0
    model_args['data'] = mx.nd.array(img_arr, ctx)
    exector = neural_net.bind(ctx, model_args ,args_grad=None, grad_req="null", aux_states=model_auxs)
    exector.forward(is_train=False)
    exector.outputs[0].wait_to_read()
    output = exector.outputs[0].asnumpy()
    return output[0]

