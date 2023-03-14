from keras.layers import Input
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
import cv2
import tensorflow as tf
import os
import time
import argparse
from tensorflow.keras.optimizers import Adam
from model import load,vgg_layers,StyleContentModel,clip_0_1,saveimg,content_layers,style_layers,style_content_loss
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

ap = argparse.ArgumentParser()
ap.add_argument("-style", "--style", type=str,required=True,
help="path to style image")
ap.add_argument("-content", "--content", type=str,required=True,
help="path to content image")
ap.add_argument("-epoch", "--epoch", type=int,default=100,
help="number epoch want to run")
ap.add_argument("-out", "--output", type=str,default='output',
help="path to *specific* model checkpoint to load")
args = vars(ap.parse_args())

style_image = load(args['style'])
content_image = load(args['content'])
extractor = StyleContentModel(style_layers, content_layers)
results = extractor(content_image)
style_targets = extractor(style_image)['style'] 
content_targets = extractor(content_image)['content']
image = tf.Variable(content_image)
opt = Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1) 


@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
      
    outputs = extractor(image)
    loss = style_content_loss(outputs,style_targets,content_targets)
    loss += 30*tf.image.total_variation(image)
  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

start = time.time()

for n in range(args['epoch']):
    print("Train step: {}".format(n+1))
    train_step(image)
    saveimg(image.numpy()[0],(n+1),args['output'])
end = time.time()
print("Total time: {:.1f}".format(end-start))
