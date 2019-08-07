
import numpy as np
import os
import shutil

def seethrough(gt_dir,classes =['Latin','Korean','Chinese','Japanese']):
    '''
    input = classes
    output =  numpy array of shape (10000,num_classes), num_classes = 4
    one image -> [Latin box 갯수, Korean box 갯수, Chinese box 갯수, Japanese box 갯수]
    '''
    textplace = gt_dir
    result = []
    num_classes = len(classes)
    for name in os.listdir(textplace)[0:10000]:
        boxes = [0]*num_classes
        txt = open(os.path.join(textplace,name), mode = 'r', encoding = 'utf-8')
        txt = txt.readlines()
        txt = [a.split(',')[8] for a in txt]
        
        for j in range(len(txt)):
            for i in range(num_classes):
                if txt[j]==classes[i]:
                    boxes[i]+=1
                    
        result.append(boxes)
        
    return np.array(result)

def make_batch_idx(gt_dir,size = 1000,except_list = None):
    
    '''
    output =  size만큼 language ratio를 맞춰 뽑아낸 image의 index들
    
    '''
    random_seed = 42
    num_classes = 4
    lang_ratio = [0.7, 0.1, 0.1, 0.1]
    
    Latin_size = int(size * lang_ratio[0])
    Korean_size = int(size * lang_ratio[1])
    Chinese_size = int(size * lang_ratio[2])
    Japanese_size = size - Latin_size - Korean_size - Chinese_size
    
    gtboxes = seethrough(gt_dir)
    list_ = [[],[],[],[]]
    for i in range(len(gtboxes)):
        for j in range(num_classes):
            if gtboxes[i][j]>0:
                list_[j].append(i)
    Latin_list = list_[0]
    Korean_list = list_[1]
    Chinese_list = list_[2]
    Japanese_list = list_[3]
    
    if except_list:
        picked_set = set(except_list)
    else:        
        picked_set = set()
    
    Latin_list = list(set(Latin_list)-picked_set)
    np.random.seed(random_seed)
    np.random.shuffle(Latin_list)
    Latin_idx = np.array(Latin_list)[np.random.permutation(Latin_size)]
    picked_set = picked_set.union(set(Latin_idx))
    
    Korean_list = list(set(Korean_list)-picked_set)
    np.random.seed(random_seed)
    np.random.shuffle(Korean_list)
    Korean_idx = np.array(Korean_list)[np.random.permutation(Korean_size)]
    picked_set = picked_set.union(set(Korean_idx))
    
    Chinese_list = list(set(Chinese_list)-picked_set)
    np.random.seed(random_seed)
    np.random.shuffle(Chinese_list)
    Chinese_idx = np.array(Chinese_list)[np.random.permutation(Chinese_size)]
    picked_set = picked_set.union(set(Chinese_idx))
    
    Japanese_list = list(set(Japanese_list)-picked_set)
    np.random.seed(random_seed)
    np.random.shuffle(Japanese_list)
    Japanese_idx = np.array(Japanese_list)[np.random.permutation(Japanese_size)]
    picked_set = picked_set.union(set(Japanese_idx))
    Latin_idx.sort()
    Korean_idx.sort()
    Chinese_idx.sort()
    Japanese_idx.sort()
    
    if except_list:
        picked_set = picked_set - set(except_list)
    
    return picked_set,Latin_idx,Korean_idx,Chinese_idx,Japanese_idx

def save_imgs_to_dir(img_dir1,img_dir2,idx_list,directory):
    
    os.makedirs(directory)
    
    name_batch = []
    imageplace = img_dir1
    for name in os.listdir(imageplace):
        name_batch.append(name)
        
    imageplace = img_dir2
    for name in os.listdir(imageplace):
        name_batch.append(name)
    
    for idx in idx_list:
        name = name_batch[idx]
        if idx+1<=5000:
            imageplace = img_dir1
        else:
            imageplace = img_dir2
        
        shutil.copy2(os.path.join(imageplace,name), os.path.join(directory,name))
    
def save_txt_to_dir(gt_dir,idx_list, directory):
    os.makedirs(directory)
    
    name_batch = []
    txtplace = gt_dir
    for name in os.listdir(txtplace):
        name_batch.append(name)        
    
    for idx in idx_list:
        name = name_batch[idx]
        shutil.copy2(os.path.join(txtplace,name), os.path.join(directory,name))

