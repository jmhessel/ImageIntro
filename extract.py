from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Flatten
import os, sys, numpy as np


def load_images(image_list):
    images = []
    for i in image_list:
        c_img = np.expand_dims(image.img_to_array(image.load_img(i, target_size = (224, 224))), axis = 0)
        images.append(c_img)
    return preprocess_input(np.vstack(images))

def image_generator(fnames, batch_size):
    while True:#Keras generators need to loop forever for some reason...
        cfns = []
        for i, p in enumerate(fnames):
            cfns.append(p)
            if len(cfns) == batch_size:
                yield load_images(cfns)
                cfns = []
        if len(cfns) != 0:
            yield load_images(cfns)
            cfns = []

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: [image list]")
        quit()

    file_list = sys.argv[1]
    fpath = "res.txt"

    all_paths = []
    with open(file_list) as f:
        for i,line in enumerate(f):
            all_paths.append(line.strip())
            
    print("Extracting from {} files".format(len(all_paths)))
            
    print('Saving to {}'.format(fpath))
    base_model = ResNet50()
    base_model.trainable = False
    image_model = Sequential()

    image_model.add(Model(input = base_model.input,
                          output=base_model.get_layer('avg_pool').output))
    image_model.add(Flatten())

    image_model.summary()
    gen = image_generator(all_paths, 32)
    feats = image_model.predict_generator(gen, len(all_paths))
    print("Saving to {}".format(fpath))
    with open(fpath, 'w') as f:
        for i in range(feats.shape[0]):
            f.write(" ".join(["{:.8f}".format(x) for x in feats[i,:]]) + "\n")

