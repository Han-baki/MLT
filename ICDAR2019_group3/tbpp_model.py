#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model

from utils.layers import Normalize

def TBPP512(input_shape=(512, 512, 3), softmax=True):
    
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd512_body(x)
    
    num_maps = len(source_layers)
    
    num_priors = [14] * num_maps  # 91
    normalizations = [1] * num_maps
    output_tensor = multibox_head(source_layers, num_priors, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    
    model.aspect_ratios = [[1,2,3,5,1/2,1/3,1/5] * 2] * num_maps
    model.shifts = [[(0.0, -0.5)] * 7 + [(0.0, 0.5)] * 7] * num_maps
    model.special_ssd_boxes = False
    model.scale = 0.5
    
    return model

def ssd512_body(x):
    
    source_layers = []
    #수정 한상준4 MaxPool2D => MaxPooling2D
    # Block 1
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_1', activation='relu')(x) #shape : (None, 512, 512, 64)
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_2', activation='relu')(x) #shape : (None, 512, 512, 64)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool1')(x) # shape: (None, 256, 256, 64)
    # Block 2
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_1', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_2', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool2')(x) # shape: (None, 128, 128, 128)
    # Block 3
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_2', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_3', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool3')(x) # shape: (None, 64, 64, 256)
    # Block 4
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_3', activation='relu')(x)
    source_layers.append(x) # shape: (None, 64, 64, 512)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool4')(x) # shape: (None, 32, 32, 512)
    # Block 5
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_3', activation='relu')(x)
    x = MaxPooling2D(pool_size=3, strides=1, padding='same', name='pool5')(x) # shape: (None, 32, 32, 512)
    # FC6
    x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x) # shape: (None, 32, 32, 1024)
    # FC7
    x = Conv2D(1024, 1, strides=1, padding='same', name='fc7', activation='relu')(x) # shape: (None, 32, 32, 1024)
    source_layers.append(x) # shape: (None, 32, 32, 1024)
    # Block 6
    x = Conv2D(256, 1, strides=1, padding='same', name='conv6_1', activation='relu')(x) # shape: (None, 32, 32, 256)
    x = ZeroPadding2D((1,1))(x) # shape: (None, 33, 33, 256)
    x = Conv2D(512, 3, strides=2, padding='valid', name='conv6_2', activation='relu')(x) # shape: (None, 16, 16, 512)
    source_layers.append(x) # shape: (None, 16, 16, 512)
    # Block 7
    x = Conv2D(128, 1, strides=1, padding='same', name='conv7_1', activation='relu')(x) # shape: (None, 16, 16, 128)
    x = ZeroPadding2D((1,1))(x) # shape: (None, 17, 17, 128)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv7_2', activation='relu')(x) # shape: (None, 8, 8, 256)
    source_layers.append(x) # shape: (None, 8, 8, 256)
    # Block 8
    x = Conv2D(128, 1, strides=1, padding='same', name='conv8_1', activation='relu')(x) # shape: (None, 8, 8, 128)
    x = ZeroPadding2D((1,1))(x) # shape: (None, 9, 9, 128)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv8_2', activation='relu')(x) # shape: (None, 4, 4, 256)
    source_layers.append(x) # shape: (None, 4, 4, 256)
    # Block 9
    x = Conv2D(128, 1, strides=1, padding='same', name='conv9_1', activation='relu')(x) # shape: (None, 4, 4, 128)
    x = ZeroPadding2D((1,1))(x) # shape: (None, 5, 5, 128)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv9_2', activation='relu')(x) # shape: (None, 2, 2, 256)
    source_layers.append(x) # shape: (None, 2, 2, 256)
    # Block 10 
    x = Conv2D(128, 1, strides=1, padding='same', name='conv10_1', activation='relu')(x) # shape: (None, 2, 2, 128)
    x = ZeroPadding2D((1,1))(x) # shape: (None, 3, 3, 128)
    x = Conv2D(256, 4, strides=2, padding='valid', name='conv10_2', activation='relu')(x) # shape: (None, 1, 1, 256)
    source_layers.append(x)  # shape: (None, 1, 1, 256)
    
    return source_layers

def multibox_head(source_layers, num_priors ,normalizations=None , softmax=True):
    
    num_classes = 6
    class_activation = 'softmax' if softmax else 'sigmoid'

    mbox_conf = []
    mbox_loc = []
    mbox_quad = []
    mbox_rbox = []
    for i in range(len(source_layers)):
        x = source_layers[i]             #i=0 => image size of x : (64,64), shape of x : [None, 64, 64, 512]
        name = x.name.split('/')[0]
        
        # normalize
        if normalizations is not None and normalizations[i] > 0:
            name = name + '_norm'
            x = Normalize(normalizations[i], name=name)(x)
            
        # confidence
        name1 = name + '_mbox_conf'
        x1 = Conv2D(num_priors[i] * num_classes, (3, 5), padding='same', name=name1)(x) #num_priors[i]=14, num_classes = 6 => x1 shape : [None, 64, 64, 14*6]
        x1 = Flatten(name=name1+'_flat')(x1) # shape of x1: [batch_size, 64*64*14*6], per class : 64*64*14
        mbox_conf.append(x1)

        # location, Delta(x,y,w,h)
        name2 = name + '_mbox_loc'
        x2 = Conv2D(num_priors[i] * 4, (3, 5), padding='same', name=name2)(x)
        x2 = Flatten(name=name2+'_flat')(x2)
        mbox_loc.append(x2)  # shape of x2: [batch_size, 64*64*14*4]
        
        # quadrilateral, Delta(x1,y1,x2,y2,x3,y3,x4,y4)
        name3 = name + '_mbox_quad'
        x3 = Conv2D(num_priors[i] * 8, (3, 5), padding='same', name=name3)(x)
        x3 = Flatten(name=name3+'_flat')(x3)
        mbox_quad.append(x3) # shape of x3: [batch_size, 64*64*14*8]

        # rotated rectangle, Delta(x1,y1,x2,y2,h)
        name4 = name + '_mbox_rbox'
        x4 = Conv2D(num_priors[i] * 5, (3, 5), padding='same', name=name4)(x)
        x4 = Flatten(name=name4+'_flat')(x4)
        mbox_rbox.append(x4) # shape of x4: [batch_size, 64*64*14*5]
        
    mbox_conf = concatenate(mbox_conf, axis=1, name='mbox_conf')
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf) #shape : [batch_size, 64*64*14, 6]
    mbox_conf = Activation(class_activation, name='mbox_conf_final')(mbox_conf)
    
    mbox_loc = concatenate(mbox_loc, axis=1, name='mbox_loc')
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc) #shape : [batch_size, 64*64*14, 4]
    
    mbox_quad = concatenate(mbox_quad, axis=1, name='mbox_quad')
    mbox_quad = Reshape((-1, 8), name='mbox_quad_final')(mbox_quad) #shape : [batch_size, 64*64*14, 8]
    
    mbox_rbox = concatenate(mbox_rbox, axis=1, name='mbox_rbox')
    mbox_rbox = Reshape((-1, 5), name='mbox_rbox_final')(mbox_rbox) #shape : [batch_size, 64*64*14, 5]

    predictions = concatenate([mbox_loc, mbox_quad, mbox_rbox, mbox_conf], axis=2, name='predictions')
    
    return predictions # shape : (batch_size,Nums_priors,4+8+5+6) = (batch_size, 76454, 23)

