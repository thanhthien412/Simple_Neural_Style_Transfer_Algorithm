from keras.applications.vgg19 import VGG19
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import os
import cv2
content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

style_weight=1e1
content_weight=1e3

def load(path_to_img):
  max_dim = 700
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def vgg_layers(layer_names):
  vgg = VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  
  return model

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


class StyleContentModel(Model):
    def __init__(self, stylelayers,contentlayer):
        super(StyleContentModel, self).__init__()
        self.vgg            =vgg_layers(stylelayers+contentlayer)
        self.style_layers   =stylelayers
        self.content_layer  =contentlayer
        self.num_style_layer=len(self.style_layers)
        self.vgg.trainable  =False
        
        
    def call(self,inputs):
        inputs = inputs*255.0
        processedinput  =preprocess_input(inputs)
        outputs         =self.vgg(processedinput)
        
        style_outputs,content_output=(outputs[:self.num_style_layer],outputs[self.num_style_layer:])
        
        style_outputs=[gram_matrix(style) for style in style_outputs]

        content_dict    ={content_name:value for content_name,value in zip(self.content_layer,content_output)}
                         
        style_dict      ={style_name:value for style_name,value in zip(self.style_layers,style_outputs)}
        
        return {'content':content_dict,'style':style_dict}
    
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    
def saveimg(image,step,output):
    cv2.imwrite(os.path.join(output,'epoch_{}.png'.format(step+1)),image*255)

def style_content_loss(outputs,style_targets,content_targets):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / 5

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / 1
    loss = style_loss + content_loss
    return loss