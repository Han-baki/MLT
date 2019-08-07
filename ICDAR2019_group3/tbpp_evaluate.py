#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from shapely.geometry import Polygon
from shapely.geometry import box as Box
from shapely.ops import unary_union
import numpy as np

def iou2(input1,input2):
    scale = 5000
    coords1 = np.array([int(scale*x) for x in input1[:8]]).reshape((-1,2))
    coords2 = np.array([int(scale*x) for x in input2[:8]]).reshape((-1,2))
    polygon1 = Polygon(coords1)
    polygon2 = Polygon(coords2)
    try : 
        _intersection = polygon1.intersection(polygon2).area
            
    except:

        _intersection= 0
        return 0
    if _intersection == 0:
        return 0
    return _intersection / unary_union([polygon1, polygon2]).area

def prh_txt(gt, pred): #precision, recall, harmonic_mean
    '''
    text detection만 따진 precision, recall, h mean 값
    '''
    score = 0
    gtb_len = 0
    predb_len = 0
    
    for image in range(len(gt)):
        mask_gt_others = (gt[image][:,8] != 5)
        gt[image] = gt[image][mask_gt_others]
        mask_pred_others = (pred[image][:,8] != 5)
        pred[image] = pred[image][mask_pred_others]
        
#         print(np.shape(gt[image]))
        gtb = gt[image][:,:8]
#         print('gtb:',np.shape(gtb))
        gtb_len+=len(gtb)
#         print('pred[image]:',np.shape(pred[image]))
        predb = pred[image][:,:8]
        predb_len+=len(predb)
        
        if len(predb)==0 or len(gtb)==0:
            continue
        
        gtb_lang = gt[image][:,8]
        predb_lang = pred[image][:,8]
        
        
            

        iou_matrix = np.array([[iou2(g,d) for g in gtb] for d in predb]) # each row -> pred, each column -> gt

        while True:
            ind = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
            if iou_matrix[ind] > 0.5:
                
                score += 1 #수정 한상준
                iou_matrix[ind[0],:] = 0
                iou_matrix[:, ind[1]] = 0
            else:
                break
    precision = 0
    recall = 0
    if gtb_len==0 or predb_len==0:
        if gtb_len ==0 and predb_len==0:
            h=1
        else:
            h=0
    else:
        precision, recall = score/predb_len, score/gtb_len
        
    if precision + recall == 0:
        h = 0
    else:
        h = 2*precision*recall/(precision+recall)
    return precision, recall, h


def prh(gt, pred): #precision, recall, harmonic_mean
    score = 0
    gtb_len = 0
    predb_len = 0
    
    for image in range(len(gt)):
        mask_gt_others = (gt[image][:,8] != 5)
        gt[image] = gt[image][mask_gt_others]
        mask_pred_others = (pred[image][:,8] != 5)
        pred[image] = pred[image][mask_pred_others]
        
#         print(np.shape(gt[image]))
        gtb = gt[image][:,:8]
#         print('gtb:',np.shape(gtb))
        gtb_len+=len(gtb)
#         print('pred[image]:',np.shape(pred[image]))
        predb = pred[image][:,:8]
        predb_len+=len(predb)
        
        if len(predb)==0 or len(gtb)==0:
            continue
        
        gtb_lang = gt[image][:,8]
        predb_lang = pred[image][:,8]

        iou_matrix = np.array([[iou2(g,d) for g in gtb] for d in predb]) # each row -> pred, each column -> gt

        while True:
            ind = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
            if iou_matrix[ind] > 0.5:
                if predb_lang[ind[0]] == gtb_lang[ind[1]] :
                    score += 1
                iou_matrix[ind[0],:] = 0
                iou_matrix[:, ind[1]] = 0
            else:
                break
    precision = 0
    recall = 0
    if gtb_len==0 or predb_len==0:
        if gtb_len ==0 and predb_len==0:
            h=1
        else:
            h=0
    else:
        precision, recall = score/predb_len, score/gtb_len
        
    if precision + recall == 0:
        h = 0
    else:
        h = 2*precision*recall/(precision+recall)
    return precision, recall, h

