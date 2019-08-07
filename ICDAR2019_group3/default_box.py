#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from utils.bboxes import polygon_to_rbox3
from utils.vis import to_rec

def iou(box, priors):
    
    # compute intersection
    inter_upleft = np.maximum(priors[:, :2], box[:2])
    inter_botright = np.minimum(priors[:, 2:4], box[2:])
    inter_wh = inter_botright - inter_upleft
    inter_wh = np.maximum(inter_wh, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    # compute union
    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (priors[:, 2] - priors[:, 0]) * (priors[:, 3] - priors[:, 1])
    union = area_pred + area_gt - inter
    # compute iou
    iou = inter / union
    
    return iou  #수정 한상준2

def non_maximum_suppression_slow(boxes, confs, iou_threshold, top_k):
    """Does None-Maximum Suppresion on detection results.
    
    Intuitive but slow as hell!!!
    
    # Agruments
        boxes: Array of bounding boxes (boxes, xmin + ymin + xmax + ymax).
        confs: Array of corresponding confidenc values.
        iou_threshold: Intersection over union threshold used for comparing 
            overlapping boxes.
        top_k: Maximum number of returned indices.
    
    # Return
        List of remaining indices.
    """
    idxs = np.argsort(-confs)
    selected = []
    for idx in idxs:
        if np.any(iou(boxes[idx], boxes[selected]) >= iou_threshold):
            continue
        selected.append(idx)
        if len(selected) >= top_k:
            break
    return selected #수정 박재현3

def non_maximum_suppression(boxes, confs, overlap_threshold, top_k):
    """Does None-Maximum Suppresion on detection results.
    
    # Agruments
        boxes: Array of bounding boxes (boxes, xmin + ymin + xmax + ymax).
        confs: Array of corresponding confidenc values.
        overlap_threshold: 
        top_k: Maximum number of returned indices.
    
    # Return
        List of remaining indices.
    
    # References
        - Girshick, R. B. and Felzenszwalb, P. F. and McAllester, D.
          [Discriminatively Trained Deformable Part Models, Release 5](http://people.cs.uchicago.edu/~rbg/latent-release5/)
    """
    eps = 1e-15
    
    boxes = boxes.astype(np.float64)

    pick = []
    x1, y1, x2, y2 = boxes.T
    
    idxs = np.argsort(confs)
    area = (x2 - x1) * (y2 - y1)
    
    while len(idxs) > 0:
        i = idxs[-1]
        
        pick.append(i)
        if len(pick) >= top_k:
            break
        
        idxs = idxs[:-1]
        
        xx1 = np.maximum(x1[i], x1[idxs])
        yy1 = np.maximum(y1[i], y1[idxs])
        xx2 = np.minimum(x2[i], x2[idxs])
        yy2 = np.minimum(y2[i], y2[idxs])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        I = w * h 
        
        overlap = I / (area[idxs] + eps)
        # as in Girshick et. al.
        
        #U = area[idxs] + area[i] - I
        #overlap = I / (U + eps)
        
        idxs = idxs[overlap <= overlap_threshold]
        
    return pick #수정 박재현3
  

def non_max_suppression_class(input_boxes,iou_threshold = 0.5): #수정 한상준8
    '''
    서로 다른 class에 있는 box끼리도 iou threshold 0.5로 겹치는거 제외하기
    input_boxes = boxes with [bbox, quad, r_box, max_confidence, class],
                box들의 max_confidence가 큰 순서대로 배열된 numpy array
    '''
    bboxes= input_boxes[:,:4]
    i=0
    
    
    while i<len(bboxes)-1:
        mask = np.array([True]*len(bboxes))
        
        mask[i+1:] = iou(bboxes[i], bboxes[i+1:])<0.5
        bboxes = bboxes[mask]
        input_boxes = input_boxes[mask]
        i+=1
    return input_boxes
  
class PriorUtil(object):
    
    def __init__(self, model, aspect_ratios=None, shifts=None,
            minmax_sizes=None, steps=None, scale=None, clips=None, 
            special_ssd_boxes=None, ssd_assignment=None):
        
        
        source_layers_names = [l.name.split('/')[0] for l in model.source_layers]
        self.source_layers_names = source_layers_names
        
        self.model = model
        self.image_size = model.input_shape[1:3]
        
        num_maps = len(source_layers_names)
        
        aspect_ratios = model.aspect_ratios #수정 한상준1
        shifts = model.shifts #수정 한상준1
        
        if minmax_sizes is None:
            if hasattr(model, 'minmax_sizes'):
                minmax_sizes = model.minmax_sizes
            else:
                # as in equation (4)
                min_dim = np.min(self.image_size)
                min_ratio = 10 # 15
                max_ratio = 100 # 90
                s = np.linspace(min_ratio, max_ratio, num_maps+1) * min_dim / 100.
                minmax_sizes = [(round(s[i]), round(s[i+1])) for i in range(len(s)-1)]

        scale = model.scale #수정 한상준1      
        minmax_sizes = np.array(minmax_sizes) * scale
        
        steps = [None] * num_maps #수정 한상준1
        
        clips = True #수정 한상준2
        clips = [clips] * num_maps #수정 한상준1
        
        special_ssd_boxes = False
        special_ssd_boxes = [special_ssd_boxes] * num_maps #수정 한상준1
        
        ssd_assignment = True #수정 한상준1
        self.ssd_assignment = ssd_assignment
        
        self.prior_maps = []
        for i in range(num_maps):
            layer = model.get_layer(source_layers_names[i])
            map_h, map_w = map_size = layer.output_shape[1:3]
            m = PriorMap(source_layer_name=source_layers_names[i],
                         image_size=self.image_size,
                         map_size=map_size,
                         minmax_size=minmax_sizes[i],
                         variances=[0.1, 0.1, 0.2, 0.2],
                         aspect_ratios=aspect_ratios[i],
                         shift=shifts[i],
                         step=steps[i],
                         special_ssd_box=special_ssd_boxes[i],
                         clip=clips[i])
            self.prior_maps.append(m)
        self.update_priors()

        self.nms_top_k = 400
        self.nms_thresh = 0.45 #수정 박재현3
    
    def update_priors(self):
        priors_xy = []
        priors_wh = []
        priors_min_xy = []
        priors_max_xy = []
        priors_variances = []
        priors = []
        
        map_offsets = [0]
        
        # 모든 층에서 정의되는 prior map들에 대한 정보를 위에서 정의한 list들에 하나하나 append를 시킨다
        for i in range(len(self.prior_maps)): 
            m = self.prior_maps[i]
            
            # compute prior boxes
            m.compute_priors()
            
            # collect prior data
            priors_xy.append(m.priors_xy)
            priors_wh.append(m.priors_wh)
            priors_min_xy.append(m.priors_min_xy)
            priors_max_xy.append(m.priors_max_xy)
            priors_variances.append(m.priors_variances)
            priors.append(m.priors)
            map_offsets.append(map_offsets[-1]+len(m.priors))
        
        # append시킨 list를 axis=0에 대하여 concatenate시켜서 length가 모든 층의 prior map의 갯수를 전부 더한것 만큼 된다.
        self.priors_xy = np.concatenate(priors_xy, axis=0)
        self.priors_wh = np.concatenate(priors_wh, axis=0)
        self.priors_min_xy = np.concatenate(priors_min_xy, axis=0)
        self.priors_max_xy = np.concatenate(priors_max_xy, axis=0)
        self.priors_variances = np.concatenate(priors_variances, axis=0)
        self.priors = np.concatenate(priors, axis=0)
        self.map_offsets = map_offsets
        
        # normalized prior boxes (마지막으로 prior box coordinate과 여러 정보들을 image size로 나누어서 normalize시킨다.)
        image_h, image_w = self.image_size
        self.priors_xy_norm = self.priors_xy / (image_w, image_h)
        self.priors_wh_norm = self.priors_wh / (image_w, image_h)
        self.priors_min_xy_norm = self.priors_min_xy / (image_w, image_h)
        self.priors_max_xy_norm = self.priors_max_xy / (image_w, image_h)
        self.priors_norm = np.concatenate([self.priors_min_xy_norm, self.priors_max_xy_norm, self.priors_variances], axis=1)
    
    
    def encode(self, gt_data, overlap_threshold=0.5, debug=False):
        
        '''
        
        gt_data의 예시:
        image 하나당 [x1,y1, x2,y2, x3,y3, x4,y4, 6_one_hot_class]의 데이터가 box갯수만큼 있음
        
        [[0.56833333 0.20300752 0.97666667 0.33333333 0.97       0.47619048
          0.55833333 0.34586466 0.         0.         0.         0.
          0.         1.        ]
         [0.11666667 0.45864662 0.14166667 0.45614035 0.14       0.49874687
          0.115      0.50125313 0.         1.         0.         0.
          0.         0.        ]
         [0.88333333 0.47869674 0.91166667 0.48120301 0.91       0.50626566
          0.88166667 0.5037594  0.         1.         0.         0.
          0.         0.        ]
         [0.03166667 0.30075188 0.42666667 0.160401   0.44666667 0.30075188
          0.03666667 0.46867168 0.         1.         0.         0.
          0.         0.        ]
         [0.15833333 0.39598997 0.42       0.40601504 0.415      0.48120301
          0.15333333 0.47368421 0.         1.         0.         0.
          0.         0.        ]
         [0.58166667 0.41854637 0.85833333 0.4235589  0.86       0.4962406
          0.58166667 0.49373434 0.         1.         0.         0.
          0.         0.        ]
         [0.63       0.57894737 0.68333333 0.60150376 0.67666667 0.64160401
          0.62333333 0.61904762 0.         1.         0.         0.
          0.         0.        ]
         [0.46       0.71428571 0.555      0.71428571 0.555      0.74937343
          0.46       0.74937343 0.         0.         0.         0.
          0.         1.        ]]
          
          위에서 정의한 prior box들의 정보를 가지고 prior box와 gt_data를 match시켜서 
          shape = [priormap갯수(엄청큼), 4+8+5+confidence_of_class] 인 넘파이배열을 return한다.
          confidence class에서는 4+8+5의 정보가 gt_data와의 iou가 0.5를 넘는게 없다면 background class로 분류된다.
          
        ''' 
        
        '''
        array([[0.09446537, 0.64231986, 0.6947241 , 0.80873207, 0.66108891,
        1.05911375, 0.05824286, 0.92018245, 0.        , 0.        ,
        0.        , 0.        , 0.        , 1.        ]]) error찾기
        '''
        # image에 box가 없을경우
        if gt_data.shape[0] == 0:
            print('gt_data', type(gt_data), gt_data.shape)
        eps = 1e-15 #수정 한상준13
        num_classes = 6
        num_priors = self.priors.shape[0]
        
        gt_polygons = np.copy(gt_data[:,:8]) # normalized quadrilaterals
        gt_rboxes = np.array([polygon_to_rbox3(np.reshape(p, (-1,2))) for p in gt_data[:,:8]])
        
        # minimum horizontal bounding rectangles
        gt_xmin = np.min(gt_data[:,0:8:2], axis=1) #shape : [nb_boxes]
        gt_ymin = np.min(gt_data[:,1:8:2], axis=1)
        gt_xmax = np.max(gt_data[:,0:8:2], axis=1)
        gt_ymax = np.max(gt_data[:,1:8:2], axis=1)
        
        gt_boxes = self.gt_boxes = np.array([gt_xmin,gt_ymin,gt_xmax,gt_ymax]).T # shape : [nb_boxes, 4]
                                                                                 # normalized xmin, ymin, xmax, ymax
        gt_one_hot = gt_data[:,8:] # shape : [nb_boxes, 6]

        gt_iou = np.array([iou(b, self.priors_norm) for b in gt_boxes]).T 
        # shape of self.priors_norm : [Nums_priors, min_xy + max_xy+ variance]
        # b = [4], self.priors_norm = [Nums_priors, 4+4]
        # shape of gt_iou : [[Nums_priors] for b in gt_boxes].transpose = [nb_boxes, Nums_priors].transpose = [Num_priors, nb_boxes]
        
        # assigne gt to priors
        max_idxs = np.argmax(gt_iou, axis=1) # shape : [Num_priors]
        max_val = gt_iou[np.arange(num_priors), max_idxs] # shape: [Num_priors]
        prior_mask = max_val > overlap_threshold #IOU값의 maximum이 0.5가 넘는 prior들만 True값을 줌 
        match_indices = max_idxs[prior_mask] # shape : [Num_priors - False_priors]

        self.match_indices = dict(zip(list(np.ix_(prior_mask)[0]), list(match_indices))) 
        # {prior_1 : max_idx, prior_3 : max_idx, prior_4 : max_idx, prior_7 : max_idx, ...}
        
        # prior labels
        confidence = np.zeros((num_priors, num_classes))
        confidence[:,0] = 1  # 일단 모든 prior들을 background 로 정의
        confidence[prior_mask] = gt_one_hot[match_indices]  # mask가 True인것(iou>0.5)만 one_hot_class를 새로 매겨줌

        gt_xy = (gt_boxes[:,2:4] + gt_boxes[:,0:2]) / 2.  #shape : [nb_boxes,2]
        gt_wh = gt_boxes[:,2:4] - gt_boxes[:,0:2]    #shape : [nb_boxes,2]
        gt_xy = gt_xy[match_indices]    #shape : [True_priors,2] , True_priors = Num_priors - False_priors
        gt_wh = gt_wh[match_indices]    #shape : [True_priors,2]

        gt_polygons = gt_polygons[match_indices]  #shape : [True_priors,8]
        gt_rboxes = gt_rboxes[match_indices]  #shape : [True_priors,5]
        
        priors_xy = self.priors_xy[prior_mask] / self.image_size  # = self.priors_xy_norm[prior_mask]
        priors_wh = self.priors_wh[prior_mask] / self.image_size  # = self.priors_wh_norm[prior_mask]
        variances_xy = self.priors_variances[prior_mask,0:2]
        variances_wh = self.priors_variances[prior_mask,2:4]
        
        # compute local offsets for 
        # gt_x = prior_x + prior_w * x_label
        # x_label = x_label / 0.1
        # gt_w = exp(w_label) * prior_w
        # w_label = w_label / 0.2
        offsets = np.zeros((num_priors, 4))
        offsets[prior_mask,0:2] = (gt_xy - priors_xy) / priors_wh
        offsets[prior_mask,2:4] = np.log(gt_wh / priors_wh )#수정 한상준13
        offsets[prior_mask,0:2] /= variances_xy
        offsets[prior_mask,2:4] /= variances_wh
        
        # compute local offsets for quadrilaterals
        # gt_x1 = prior_x1 + prior_w * x1_label 
        # x1_label = x1_label / 0.1
        offsets_quads = np.zeros((num_priors, 8))
        priors_xy_minmax = np.hstack([priors_xy-priors_wh/2, priors_xy+priors_wh/2])
        ref = priors_xy_minmax[:,(0,1,2,1,2,3,0,3)] # corner points
        offsets_quads[prior_mask,:] = (gt_polygons - ref) / np.tile(priors_wh, (1,4)) / np.tile(variances_xy, (1,4))
        
        # compute local offsets for rotated bounding boxes
        # gt_x1 = prior_x1 + prior_w * x1_label
        # x1_label = x1_label / 0.1
        # gt_h = exp(h_label) * prior_h
        # h_label = h_label / 0.2
        offsets_rboxs = np.zeros((num_priors, 5))
        offsets_rboxs[prior_mask,0:2] = (gt_rboxes[:,0:2] - priors_xy) / priors_wh / variances_xy
        offsets_rboxs[prior_mask,2:4] = (gt_rboxes[:,2:4] - priors_xy) / priors_wh / variances_xy
        offsets_rboxs[prior_mask,4] = np.log(gt_rboxes[:,4] / priors_wh[:,1] ) / variances_wh[:,1] #수정 한상준13
        
        return np.concatenate([offsets, offsets_quads, offsets_rboxs, confidence], axis=1)
      
      
    def decode(self, model_output, confidence_threshold=0.01, keep_top_k=200, fast_nms=False, sparse=True):
        # calculation is done with normalized sizes
        # mbox_loc, mbox_quad, mbox_rbox, mbox_conf
        # 4,8,5,2
        # boxes, quad, rboxes, confs, labels
        # 4,8,5,1,1
        
#        print("decode debugging")
#        print(model_output[:500, :])
        
        prior_mask = model_output[:,17:] > confidence_threshold
        
        if sparse:
            # compute boxes only if the confidence is high enough and the class is not background
            mask = np.any(prior_mask[:,1:], axis=1)
            prior_mask = prior_mask[mask]
            mask = np.ix_(mask)[0]
            model_output = model_output[mask]
            priors_xy = self.priors_xy[mask] / self.image_size
            priors_wh = self.priors_wh[mask] / self.image_size
            priors_variances = self.priors_variances[mask,:]
        else:
            priors_xy = self.priors_xy / self.image_size
            priors_wh = self.priors_wh / self.image_size
            priors_variances = self.priors_variances
            
        #print('offsets', len(confidence), len(prior_mask))
        
        offsets = model_output[:,:4]
        offsets_quads = model_output[:,4:12]
        offsets_rboxs = model_output[:,12:17]
        confidence = model_output[:,17:]
        
        priors_xy_minmax = np.hstack([priors_xy-priors_wh/2, priors_xy+priors_wh/2])
        ref = priors_xy_minmax[:,(0,1,2,1,2,3,0,3)] # corner points
        variances_xy = priors_variances[:,0:2]
        variances_wh = priors_variances[:,2:4]
        
        num_priors = offsets.shape[0]
        num_classes = confidence.shape[1]

        # compute bounding boxes from local offsets
        boxes = np.empty((num_priors, 4))
        offsets = offsets * priors_variances
        boxes_xy = priors_xy + offsets[:,0:2] * priors_wh
        boxes_wh = priors_wh * np.exp(offsets[:,2:4])
        boxes[:,0:2] = boxes_xy - boxes_wh / 2. # xmin, ymin
        boxes[:,2:4] = boxes_xy + boxes_wh / 2. # xmax, ymax
        boxes = np.clip(boxes, 0.0, 1.0)
        
        # do non maximum suppression
        results = []
        for c in range(1, num_classes):
            mask = prior_mask[:,c]
            boxes_to_process = boxes[mask]
            if len(boxes_to_process) > 0:
                confs_to_process = confidence[mask, c]
                
                if fast_nms:
                    idx = non_maximum_suppression(
                            boxes_to_process[:,:4], confs_to_process, 
                            self.nms_thresh, self.nms_top_k)
                else:
                    idx = non_maximum_suppression_slow(
                            boxes_to_process[:,:4], confs_to_process, 
                            self.nms_thresh, self.nms_top_k)
                
                good_boxes = boxes_to_process[idx]
                good_confs = confs_to_process[idx][:, None]
                labels = np.ones((len(idx),1)) * c
                
                good_quads = ref[mask][idx] + offsets_quads[mask][idx] * np.tile(priors_wh[mask][idx], (1,4)) * np.tile(variances_xy[mask][idx], (1,4))

                good_rboxs = offsets_rboxs[mask][idx]
                
                good_rboxs = np.empty((len(idx), 5))
                good_rboxs[:,0:2] = priors_xy[mask][idx] + offsets_rboxs[mask][idx,0:2] * priors_wh[mask][idx] * variances_xy[mask][idx]
                good_rboxs[:,2:4] = priors_xy[mask][idx] + offsets_rboxs[mask][idx,2:4] * priors_wh[mask][idx] * variances_xy[mask][idx]
                good_rboxs[:,4] = np.exp(offsets_rboxs[mask][idx,4] * variances_wh[mask][idx,1]) * priors_wh[mask][idx,1]
                
                c_pred = np.concatenate((good_boxes, good_quads, good_rboxs, good_confs, labels), axis=1)
               
                results.extend(c_pred)
        if len(results) > 0:
            results = np.array(results)
            order = np.argsort(-results[:, 17])
            results = results[order]
            results = results[:keep_top_k]
            #class끼리도 IOU threshold 0.5로 없애기
            results = non_max_suppression_class(results,iou_threshold = 0.5) #수정 한상준8
        else:
            results = np.empty((0,19))#수정 한상준9
        self.results = results
        return results    #수정 박재현3

    def plot_results(self, results=None, classes=None, show_labels=True, gt_data=None, confidence_threshold=None):
        if results is None:
            results = self.results
        if confidence_threshold is not None:
            mask = results[:, 4] > confidence_threshold
            results = results[mask]
        if classes is not None:
            colors = plt.cm.hsv(np.linspace(0, 1, len(classes)+1)).tolist()
        ax = plt.gca()
        im = plt.gci()
        image_size = im.get_size()
        
        # draw ground truth
        if gt_data is not None:
            for box in gt_data:
                label = np.nonzero(box[4:])[0][0]+1
                color = 'g' if classes == None else colors[label]
                xy_rec = to_rec(box[:4], image_size)
                ax.add_patch(plt.Polygon(xy_rec, fill=True, color=color, linewidth=1, alpha=0.3))
        
        # draw prediction
        for r in results:

            label = int(r[18])
            confidence = r[17]

            color = 'r' if classes == None else colors[label]
            xy_rec = to_rec(r[:4], image_size)
            ax.add_patch(plt.Polygon(xy_rec, fill=False, edgecolor=color, linewidth=2))
            if show_labels:
                label_name = label if classes == None else classes[label]
                xmin, ymin = xy_rec[0]
                display_txt = '%0.2f, %s' % (confidence, label_name)        
                ax.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5}) #수정 박재현4

class PriorMap(object):
    
    def __init__(self, source_layer_name, image_size, map_size, 
                 minmax_size=None, variances=[0.1, 0.1, 0.2, 0.2], 
                 aspect_ratios=[1], shift=None,
                 clip=False, step=None, special_ssd_box=False):
        self.__dict__.update(locals())  #수정 한상준3
#         self.compute_priors()

    def compute_priors(self):
        image_h, image_w = image_size = self.image_size
        map_h, map_w = map_size = self.map_size
        min_size, max_size = self.minmax_size
        # map_size = [(64,64),(32,32),(16,16),(8,8),(4,4),(2,2),(1,1)]
        
        # define centers of prior boxes
        if self.step is None:
            step_x = round(image_w / map_w)
            step_y = round(image_h / map_h)
        else:
            step_x = step_y = self.step
        # step_x, step_y = [(8,8), (16,16), (32,32), (64,64), (128,128), (256,256), (512,512)]
            
        linx = np.array([(0.5 + i) for i in range(map_w)]) * step_x
        liny = np.array([(0.5 + i) for i in range(map_h)]) * step_y
        box_xy = np.array(np.meshgrid(linx, liny)).reshape(2,-1).T
        
        if self.shift is None:
            shift = [(0.0,0.0)] * len(self.aspect_ratios)
        else:
            shift = self.shift
        
        box_wh = []
        box_shift = []
        for i in range(len(self.aspect_ratios)):
            ar = self.aspect_ratios[i]
            box_wh.append([min_size * np.sqrt(ar), min_size / np.sqrt(ar)])
            box_shift.append(shift[i])
            
        box_wh = np.asarray(box_wh)
        
        box_shift = np.asarray(box_shift)
        box_shift = np.clip(box_shift, -1.0, 1.0)
        box_shift = box_shift * 0.5 * np.array([step_x, step_y]) # percent to pixels
        
        # values for individual prior boxes
        priors_shift = np.tile(box_shift, (len(box_xy),1))
        priors_xy = np.repeat(box_xy, len(box_wh), axis=0) + priors_shift
        priors_wh = np.tile(box_wh, (len(box_xy),1))
                
        priors_min_xy = priors_xy - priors_wh / 2.
        priors_max_xy = priors_xy + priors_wh / 2.
        
        if self.clip: #수정 한상준2 들여쓰기오류
            priors_min_xy[:,0] = np.clip(priors_min_xy[:,0], 0, image_w)
            priors_min_xy[:,1] = np.clip(priors_min_xy[:,1], 0, image_h)
            priors_max_xy[:,0] = np.clip(priors_max_xy[:,0], 0, image_w)
            priors_max_xy[:,1] = np.clip(priors_max_xy[:,1], 0, image_h)
        
        priors_variances = np.tile(self.variances, (len(priors_xy),1))
        
        self.box_xy = box_xy
        self.box_wh = box_wh
        self.box_shfit = box_shift
        
        self.priors_xy = priors_xy
        self.priors_wh = priors_wh
        self.priors_min_xy = priors_min_xy
        self.priors_max_xy = priors_max_xy
        self.priors_variances = priors_variances
        self.priors = np.concatenate([priors_min_xy, priors_max_xy, priors_variances], axis=1)

