#coding=utf-8
import numpy as np
import random
import cPickle
from PIL import Image

# This is a data helper of processing cifar10.
# You can use this module to load file and feed into CNN
def unpickle(file):
    with open(file, 'rb') as fo:
        data = cPickle.load(fo)
    return data
"""
b'batch_label': filename
b'labels': a list of [1*10000]
b'data': 10000*3072
b'filename': jpgname
"""


class CifarLoader(object):
    def __init__(self, cifar_ab_path, phase, batch_size, num_epoches, shuff):
        self.batch_size = batch_size
        self.epoches = num_epoches
        self.shuffle = shuff

        filename = []
        if phase == 'TEST':
            filename.append(cifar_ab_path + "test_batch")
            self.datum_labels = []
            self.datum_images = np.zeros((10000, 3, 32, 32))
        elif phase == "TRAIN":
            for i in range(1, 6):
                filename.append(cifar_ab_path + "data_batch_%d"%i)
            self.datum_labels = []
            self.datum_images = np.zeros((50000, 3, 32, 32))

        for i, item in enumerate(filename):
            data = unpickle(item)
            images, labels = data[b'data'], list(data[b'labels'])
            low, high = i * 10000, (i+1)*10000
            self.datum_labels.extend(labels)
            images = np.reshape(images, (-1, 3, 32, 32))
            self.datum_images[low:high] = images
        print self.datum_images[0, 0, 0, 0:10]
    
    def next_batch(self):
        batch_size, epoches = self.batch_size, self.epoches
        length = len(self.datum_labels)
        indices = range(length)
        if self.shuffle == True:
            random.shuffle(indices)

        batch_images = np.zeros((batch_size, 3, 32, 32))
        batch_labels = np.zeros((batch_size, 10))
        ibatch = 0

        for epoch in range(epoches):
            for index in indices:
                for i in range(10):
                    if i == self.datum_labels[index]:
                        batch_labels[ibatch, i] = 1
                    else:
                        batch_labels[ibatch, i] = 0
                        
                batch_images[ibatch] = self.datum_images[index]
                ibatch += 1
                if ibatch == batch_size:
                    yield batch_images, batch_labels
                    ibatch = 0


"""  self testing part: saving ndarray to image!
cifar = CifarLoader('/mnt/storage01/liucan/coeus_test/cifar10/', "TEST", 1000, 1, False)
a = cifar.next_batch()
x = 0
for i,j in a:
    im = i[0].swapaxes(0,1).swapaxes(1,2)
    print im.shape
    img = Image.fromarray(im.astype('uint8')).convert('RGB')
    name = '%d.jpg'%x
    img.save(name)
    x += 1
"""
