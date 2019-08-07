#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import tensorflow as tf
import keras.backend as K

def smooth_l1_loss(y_true, y_pred):
    
    abs_loss = tf.abs(y_true - y_pred)
    sq_loss = 0.5 * (y_true - y_pred)**2
    loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    return tf.reduce_sum(loss, axis=-1)

def focal_loss(y_true, y_pred, gamma=2., alpha=1.):
    

    eps = K.epsilon()
    y_pred = K.clip(y_pred, 1e-7, 1- 1e-7)
    pt = tf.where(tf.equal(y_true, 1.), y_pred, 1.-y_pred)
    loss = - K.pow(1.-pt, gamma) * K.log(pt)   #수정 한상준10
    loss = alpha * loss
    

    return tf.reduce_sum(loss, axis=-1)


class TBPPFocalLoss(object):

    def __init__(self, lambda_conf=1000.0, lambda_offsets=0.1):
        self.lambda_conf = lambda_conf
        self.lambda_offsets = lambda_offsets
        self.metrics = []
    
    def compute(self, y_true, y_pred):
        # y.shape (batches, priors, 4 x bbox_offset + 8 x quadrilaterals + 5 x rbbox_offsets + n x class_label)
        
        
        batch_size = tf.shape(y_true)[0]
        num_priors = tf.shape(y_true)[1]
        num_classes = tf.shape(y_true)[2] - 17
        eps = K.epsilon() #keras.backend.epsilon() 1e-07
        
        # confidence loss
        conf_true = tf.reshape(y_true[:,:,17:], [-1, num_classes])
        conf_pred = tf.reshape(y_pred[:,:,17:], [-1, num_classes])
        
        class_true = tf.argmax(conf_true, axis=1) # shape = [batch_size*76454]
        class_pred = tf.argmax(conf_pred, axis=1) # shape = [batch_size*76454]
        conf = tf.reduce_max(conf_pred, axis=1) # shape = [batch_size*76454]
        
        neg_mask_float = conf_true[:,0] # background box들을 mask함
        neg_mask = tf.cast(neg_mask_float, tf.bool) # 1 => True, 0 => False
        pos_mask = tf.logical_not(neg_mask) #True => False, False => True
        pos_mask_float = tf.cast(pos_mask, tf.float32) #True => 1, False => 0
        num_total = tf.cast(tf.shape(conf_true)[0], tf.float32)
        num_pos = tf.reduce_sum(pos_mask_float)
        num_neg = num_total - num_pos
        
        conf_loss = focal_loss(conf_true, conf_pred) # 수정 한상준5 alpha삭제
        conf_loss = tf.reduce_sum(conf_loss)
        
        conf_loss = conf_loss / (num_total + eps)
        
        # offset loss, bbox, quadrilaterals, rbbox
        loc_true = tf.reshape(y_true[:,:,0:17], [-1, 17])
        loc_pred = tf.reshape(y_pred[:,:,0:17], [-1, 17])
        
        loc_loss = smooth_l1_loss(loc_true, loc_pred)
        #loc_loss = smooth_l1_loss(loc_true[:,:4], loc_pred[:,:4])
        pos_loc_loss = tf.reduce_sum(loc_loss * pos_mask_float) # only for positives
        
        loc_loss = pos_loc_loss / (num_pos + eps)
        
        # total loss
        total_loss = self.lambda_conf * conf_loss + self.lambda_offsets * loc_loss

        return total_loss

