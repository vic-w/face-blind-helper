#coding=utf-8

import os
import tornado.ioloop
import tornado.web
import model
import logging
import sys

logging.basicConfig(format='%(asctime)s %(message)s', stream=sys.stdout)

class SimilarityHandler(tornado.web.RequestHandler):
    
    def post(self):
        logging.info('request')
        upload_path='./logs/history_images'
        file1=self.request.files['image1'][0]['filename']
        file2=self.request.files['image2'][0]['filename']

        basename1, ext1 = os.path.splitext(file1)
        basename2, ext2 = os.path.splitext(file2)
        
        if (ext1 != '.jpg' and ext1 != '.JPG') or (ext2 != '.jpg' and ext2 != '.JPG'):
            return 0
        
        with open('image1.jpg','wb') as up:
            up.write(self.request.files['image1'][0]['body'])
        with open('image2.jpg','wb') as up:
            up.write(self.request.files['image2'][0]['body'])

        feature1 = model.align('image1.jpg')
        feature2 = model.align('image2.jpg')
        
        similarity = model.compare(feature1, feature2)
        
        self.write(str(similarity))


if __name__ == '__main__':
    
    logging.info('server start')
    app=tornado.web.Application([ (r'/similarity', SimilarityHandler) ])
    app.listen(3000)
    tornado.ioloop.IOLoop.instance().start()



