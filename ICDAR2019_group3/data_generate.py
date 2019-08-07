
import numpy as np
import matplotlib.pyplot as plt
import cv2 # 수정 한상준4
import random
import os
from thirdparty.get_image_size import get_image_size #수정 한상준2


class GTUtility(object):

    def __init__(self, image_path, gt_path, val = False):
        self.val = val
        self.gt_path = gt_path
        self.image_path = image_path 
        self.classes = ['Background', 'Latin', 'Chinese', 'Korean', 'Japanese', 'Others']
        one_hots = [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]]
        
        self.image_names = []
        self.data = []

        image_files = sorted(os.listdir(image_path))
        for image_file_name in image_files:
            img_width, img_height = get_image_size(os.path.join(image_path, image_file_name))
            boxes = []
            gt_file_name = os.path.splitext(image_file_name)[0] + '.txt'
            if self.val:
                gt_file_name = 'gt_'+gt_file_name
            
            with open(os.path.join(gt_path, gt_file_name), 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line_split = line.strip().split(',')
                    box = [float(v) for v in line_split[0:8]]
                    
                    classified = False
                    for lang, one_hot in zip(self.classes, one_hots):
                        if line_split[8] == lang:
                            box = box + one_hot
                            classified = True # 수정 박재현2
                    if not classified: #분류가 안된 box들은 'Others' class로 분류한다.
                        box = box + one_hots[5]
                            
                    boxes.append(box)


            boxes = np.asarray(boxes)
            boxes[:,0:8:2] /= img_width
            boxes[:,1:8:2] /= img_height
            self.image_names.append(image_file_name)
            self.data.append(boxes)
            
        self.init() # 수정 한상준2
     
    def init(self):
        self.num_classes = len(self.classes)
        self.classes_lower = [s.lower() for s in self.classes]

        # statistics
        stats = np.zeros(self.num_classes)
        num_without_annotation = 0
        for i in range(len(self.data)):

            if len(self.data[i]) == 0: # box가 없는 image
                num_without_annotation += 1
            else:
                unique, counts = np.unique(np.argmax(self.data[i][:,8:],axis = 1).astype(np.int16), return_counts=True)  # 수정 한상준3
                stats[unique] += counts
        self.stats = stats
        self.num_without_annotation = num_without_annotation
        
        self.num_samples = len(self.image_names)
        self.num_images = len(self.data)
        self.num_objects = sum(self.stats)
   
    def sample(self, idx=None): #수정 한상준9
        '''Draw a random sample form the dataset.
        '''
        if idx is None:
            idx = np.random.randint(0, len(self.image_names))
        file_path = os.path.join(self.image_path, self.image_names[idx])
        img = cv2.imread(file_path)

        img = img[:, :, (2,1,0)] # BGR to RGB
        img = img / 255.
        return idx, img, self.data[idx]
    
    def sample_random_batch(self, batch_size=32, input_size=(512,512), seed=1337): #수정 한상준9
        
        h, w = input_size
        aspect_ratio = w/h
        if seed is not None:
            np.random.seed(seed)
        idxs = np.arange(self.num_samples)
        np.random.shuffle(idxs)
        idxs = idxs[:batch_size] #수정 한상준10
        
        inputs = []
        images = []
        data = []
        
        for i in idxs:
            
            img_path = os.path.join(self.image_path, self.image_names[i])
            img = cv2.imread(img_path)
            
            gt = self.data[i]
            
            inputs.append(preprocess(img, input_size))
            img = cv2.resize(img, (w,h), cv2.INTER_LINEAR).astype('float32') # should we do resizing
            img = img[:, :, (2,1,0)] # BGR to RGB
            img /= 255
            images.append(img)
            data.append(gt)
        inputs = np.asarray(inputs)

        return idxs, inputs, images, data
    
    def sample_batch(self, batch_size, batch_index, input_size=(512,512)):#수정 한상준9
        h, w = input_size
        aspect_ratio = w/h
        idxs = np.arange(min(batch_size*batch_index, self.num_samples), 
                         min(batch_size*(batch_index+1), self.num_samples))
        
        if len(idxs) == 0:
            print('WARNING: empty batch')
        
        inputs = []
        data = []
        for i in idxs:
            img_path = os.path.join(self.image_path, self.image_names[i])
            img = cv2.imread(img_path)
            inputs.append(preprocess(img, input_size))
            data.append(self.data[i])
        inputs = np.asarray(inputs)
        
        return inputs, data
    
        
def preprocess(img, size): #수정 한상준3
    """Precprocess an image for ImageNet models.
    
    # Arguments
        img: Input Image
        size: Target image size (height, width).
    
    # Return
        Resized and mean subtracted BGR image, if input was also BGR.
    """
    h, w = size
    img = np.copy(img)
    img = cv2.resize(img, (w,h), cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    mean = np.array([104,117,123])
    img -= mean[np.newaxis, np.newaxis, :] 
    return img


class InputGenerator(object):
    """Model input generator for data augmentation."""
    # TODO
    # flag to protect bounding boxes from cropping?
    # padding to preserve aspect ratio? crop_area_range=[0.75, 1.25]
    
    def __init__(self, gt_util, prior_util, batch_size, input_size,
                preserve_aspect_ratio=True,
                augmentation=True, # 수정 한상준3
                saturation_var=0.5,
                brightness_var=0.5,
                contrast_var=0.5,
                lighting_std=0.5,
                hflip_prob=0.5,
                vflip_prob=0.5,
                do_crop=True,
                add_noise=False,
                crop_area_range=[0.75, 1.0],
                aspect_ratio_range=[4./3., 3./4.]):
        
        self.__dict__.update(locals()) 
        
        
        self.num_batches = gt_util.num_samples // batch_size
    
    def random_sized_crop(self, img, target):
        img_h, img_w = img.shape[:2]
        
        # make sure that we can preserve the aspect ratio
        ratio_range = self.aspect_ratio_range
        random_ratio = ratio_range[0] + np.random.random() * (ratio_range[1] - ratio_range[0])
        # a = w/h, w_i-w >= 0, h_i-h >= 0 leads to LP: max. h s.t. h <= w_i/a, h <= h_i
        max_h = min(img_w/random_ratio, img_h)
        max_w = max_h * random_ratio
        
        # scale the area
        crop_range = self.crop_area_range
        random_scale = crop_range[0] + np.random.random() * (crop_range[1] - crop_range[0])
        target_area = random_scale * max_w * max_h
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        x = np.random.random() * (img_w - w)
        y = np.random.random() * (img_h - h)
        
        w_rel = w / img_w
        h_rel = h / img_h
        x_rel = x / img_w
        y_rel = y / img_h
        
        w, h, x, y = int(w), int(h), int(x), int(y)
        
        # crop image and transform boxes
        new_img = img[y:y+h, x:x+w]
        num_coords = 8  # 우리의 경우는 polynomial case 수정 한상준3
        new_target = []
        if num_coords == 8: # polynom case
            for box in target:
                
                new_box = np.copy(box)
                new_box[0:8:2] -= x_rel
               
                new_box[0:8:2] /= w_rel
                
                new_box[1:8:2] -= y_rel
                
                new_box[1:8:2] /= h_rel
                

                
                if (new_box[0] < 1 and new_box[6] < 1 and new_box[2] > 0 and new_box[4] > 0 and 
                    new_box[1] < 1 and new_box[3] < 1 and new_box[5] > 0 and new_box[7] > 0):
                    
                    new_target.append(new_box)
            new_target = np.asarray(new_target)

            new_target = np.asarray(new_target).reshape(-1, target.shape[1]) # target.shape[1] = 8+num_classes = 14

        return new_img, new_target
    
    def generate(self, debug=False, encode=True, seed=None):#수정 한상준10
        h, w = self.input_size #(512, 512)
        mean = np.array([104,117,123])
        gt_util = self.gt_util
        batch_size = self.batch_size
        num_batches = self.num_batches
        aspect_ratio = w/h # 1
        
        if seed is not None:
            np.random.seed(seed)
        
        inputs, targets = [], []
        
        while True:
            idxs = np.arange(gt_util.num_samples)
            np.random.shuffle(idxs)
            idxs = idxs[:num_batches*batch_size]
            
            for j, i in enumerate(idxs):                          
                img_name = gt_util.image_names[i]
                
                img_path = os.path.join(gt_util.image_path, img_name)
                img = cv2.imread(img_path)
                y = np.copy(gt_util.data[i])

                if self.augmentation:
                    if self.do_crop:
                        for _ in range(10): # tries to crop without losing ground truth
                            img_tmp, y_tmp = self.random_sized_crop(img, y)
                            if len(y_tmp) > 0:
                                break
                        if len(y_tmp) > 0:
                            img = img_tmp
                            y = y_tmp
                    img = cv2.resize(img, (w,h), cv2.INTER_LINEAR)
                    img = img.astype(np.float32)

                    
                else:
                    img = cv2.resize(img, (w,h), cv2.INTER_LINEAR)
                    img = img.astype(np.float32)

                img -= mean[np.newaxis, np.newaxis, :] 
                
                
                inputs.append(img)
                targets.append(y)
                
                
                #if len(targets) == batch_size or j == len(idxs)-1: # last batch in epoch can be smaller then batch_size
                if len(targets) == batch_size:
                    if encode:
                        targets = [self.prior_util.encode(y) for y in targets]
                        targets = np.array(targets, dtype=np.float32)
                    tmp_inputs = np.array(inputs, dtype=np.float32)
                    tmp_targets = targets
                    inputs, targets = [], []
                    yield tmp_inputs, tmp_targets
                    
                    
                elif j == len(idxs)-1:
                    # forgett last batch
                    inputs, targets = [], []
                    break
                    
            print('NEW epoch')
        print('EXIT generator')

