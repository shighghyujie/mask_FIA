from PIL import Image 
from PIL import ImageEnhance
import numpy as np
import scipy
import torch
import torchvision.models as models
from torchvision import datasets, transforms
import feature
import stick
import predict
from heuristicsDE import differential_evolution
import os
# import xlwt
from tqdm import tqdm
import time 
import copy
import random
import rotate
# import mapping3d
import cv2
from retrieval import getLabelLists



target_class = ''
""" Perturb the image with the given pixel(s) x and get the prediction of the model """
def predict_classes(cleancrop,xs, gorb, initial_pic, target_class, searchspace, \
    sticker,opstickercv,magnification, zstore, facemask, minimize=True):
    imgs_perturbed, valid = predict.perturb_image(xs, initial_pic, \
        sticker, opstickercv, magnification, zstore, searchspace, facemask)
    # le = len(imgs_perturbed)
    # dir = './temp/'
    # for i in range(le):
    #     img_perturbed = imgs_perturbed[i]
    #     # print(dir+str(i)+'.jpg')
    #     img_perturbed.save(dir+str(i)+'.jpg')
    predictions = []
    le = len(imgs_perturbed)

    if(threat_model == 'facenet'):
        rank, pred_p = predict.predict_type_facenet(imgs_perturbed,cleancrop)
    elif(threat_model == 'arcface'):
        rank, pred_p = predict.predict_type_arcface(imgs_perturbed,cleancrop)
    elif(threat_model == 'sphereface'):
        rank, pred_p = predict.predict_type_sphereface(imgs_perturbed,cleancrop)
    elif(threat_model == 'cosface'):
        rank, pred_p = predict.predict_type_cosface(imgs_perturbed,cleancrop)
    elif(threat_model == 'yolov5'):
        rank, pred_p = predict.predict_type_yolov5(imgs_perturbed, cleancrop)
    elif(threat_model == 'taolipai'):
        rank, pred_p = predict.predict_type_taolipai(imgs_perturbed, target_class, cleancrop)
    elif(threat_model == 'retrieval'):
        rank, pred_p = predict.predict_type_retrieval(imgs_perturbed, target_class, cleancrop)
    global timess    
    timess=timess+1
    global convert, start, latter
    global generate_rank, generate_score, best_rank, best_score
    print('times = ',timess,'start = ',start,'convert = ',convert)
    '''             Remove the prob of label2
    for i in range(le):
        if(rank[i][0] != target_class):   # untarget
            probab = -1 * pinf
        else:
            label2 = rank[i][1]
            probab1 = pred_p[i][target_class].item()
            if(start == False):
                probab2 = pred_p[i][label2].item()
                a,b = 1,0
                probab = a * probab1 - b * probab2
            elif(start == True):
                probab2 = pred_p[i][latter].item()
                #a,b = 0.3,0.7
                #probab = a * probab1 - b * probab2
                beta = 20 if threat_model!='arcface' else 1
                probab = beta*(probab1 - probab2)/probab1 + (probab1 - probab2)
        if(valid[i] == 0):
            probab = pinf
        # if(probab<-99):
        #     print('ooooooo ',xs[i])
        predictions.append(np.array(probab))
    '''
    for i in range(le):
        if(rank[i] != target_class):     #untarget
            probab = -1 * pinf 
        else:
            probab = pred_p[i]
        # if(valid[i] == 0):
        #     probab = pinf
        predictions.append(np.array(probab))

    predictions = np.array(predictions)
    duplicate = copy.deepcopy(predictions)
    current_optimal = int(duplicate.argsort()[0])
    # remove the adaptive adjustment of evaluation criteria
    # mingap = pred_p[current_optimal][rank[current_optimal][0]].item() - pred_p[current_optimal][rank[current_optimal][1]].item()
    #print('mingap = ',mingap)
    '''
    if(gorb == 0):
        generate_rank.append([rank[current_optimal][0],rank[current_optimal][1]])
        generate_score.append([pred_p[current_optimal][rank[current_optimal][0]].item(),pred_p[current_optimal][rank[current_optimal][1]].item()])
        sid = int(xs[current_optimal][0])
        #print('x,y = ',int(searchspace[sid][0]),int(searchspace[sid][1]))
    elif(gorb == 1):
        best_rank.append([rank[current_optimal][0],rank[current_optimal][1]])
        best_score.append([pred_p[current_optimal][rank[current_optimal][0]].item(),pred_p[current_optimal][rank[current_optimal][1]].item()])
        sid = int(xs[current_optimal][0])
        #print('x,y = ',int(searchspace[sid][0]),int(searchspace[sid][1]))
    '''
    if(gorb == 0):
        generate_rank.append([rank[current_optimal]])
        generate_score.append([pred_p[current_optimal]])
        sid = int(xs[current_optimal][0])
        #print('x,y = ',int(searchspace[sid][0]),int(searchspace[sid][1]))
    elif(gorb == 1):
        best_rank.append([rank[current_optimal]])
        best_score.append([pred_p[current_optimal]])
        sid = int(xs[current_optimal][0])
        #print('x,y = ',int(searchspace[sid][0]),int(searchspace[sid][1]))
    '''
    if(start==False and rank[current_optimal][0] == target_class and mingap <= bound):
        start = True
        latter = rank[current_optimal][1]
        convert = True
        print('--------------convert to target attack--------')
        #print('mingap = ',mingap)
    '''
    if(start == False and rank[current_optimal] == target_class):
        # start = True
        latter = rank[current_optimal]
        # convert = True
        print('--------------convert to target attack--------')
    print(rank[current_optimal])
    print(pred_p[current_optimal])
    print(xs[current_optimal])
    return predictions, rank, convert, pred_p, valid

def convert_energy(rank, pred_p, valid, target_class):
    """
    Recalculate the fitness of the current population when adjusting the evolutionary strategy
    """
    global convert
    convert = False
    print('----------convert_energy------------')
    predictions = []
    '''
    for i in range(len(rank)):
        if(rank[i][0] != target_class):   # untarget
            probab = -1 * pinf
        else:
            label2 = rank[i][1]
            probab1 = pred_p[i][target_class].item()
            probab2 = pred_p[i][latter].item()
            #a,b = 0.3,0.7
            #probab = a * probab1 - b * probab2
            beta = 20 if threat_model!='arcface' else 1
            probab = beta*(probab1 - probab2)/probab1 + (probab1 - probab2)
        if(valid[i] == 0):
            probab = pinf
        predictions.append(np.array(probab))
    '''
    # Remove the adaptive adjustment of evaluation criteria
    for i in range(len(rank)):
        if(rank[i] != target_class):
            probab = -1 * pinf
        else:
            probab = pred_p[i]
        # if(valid[i] == 0):
        #     probab = pinf
        predictions.append(np.array(probab))
    predictions = np.array(predictions)
    return predictions    

def single_predict(cleancrop,xs, initial_pic, true_label, searchspace, \
    sticker,opstickercv,magnification, zstore, facemask):
    # imgs_perturbed, valid = predict.perturb_image(xs, initial_pic, \
    #     sticker, opstickercv, magnification, zstore, searchspace, facemask)
    imgs_perturbed, valid = predict.simple_perturb(xs, initial_pic, \
        sticker, opstickercv, searchspace, facemask)
    
    if(threat_model == 'facenet'):
        rank, pred_p = predict.predict_type_facenet(imgs_perturbed,cleancrop)
    elif(threat_model == 'arcface'):
        rank, pred_p = predict.predict_type_arcface(imgs_perturbed,cleancrop)
    elif(threat_model == 'sphereface'):
        rank, pred_p = predict.predict_type_sphereface(imgs_perturbed,cleancrop)
    elif(threat_model == 'cosface'):
        rank, pred_p = predict.predict_type_cosface(imgs_perturbed,cleancrop)
    elif(threat_model == 'yolov5'):
        rank, pred_p = predict.predict_type_yolov5(imgs_perturbed, cleancrop)
    elif(threat_model == 'taolipai'):
        rank, pred_p = predict.predict_type_taolipai(imgs_perturbed,target_class, cleancrop)
    elif(threat_model == 'retrieval'):
        rank, pred_p = predict.predict_type_retrieval(imgs_perturbed,target_class, cleancrop)
    predictions = []
    '''
    for i in range(len(imgs_perturbed)):
        if(rank[i][0] != true_label):   # untarget
            probab = -1 * pinf
        else:
            probab = pred_p[i][true_label].item()

        if(valid[i] == 0):
            probab = pinf
        predictions.append(probab)
    '''

    for i in range(len(imgs_perturbed)):
        if(rank[i] != true_label):
            probab = -1 * pinf
        else:
            probab = pred_p[i]
        # if(valid[i] == 0):
        #     probab = pinf
        predictions.append(probab)

    predictions = np.array(predictions)

    return predictions

"""  If the prediction is what we want (misclassification or targeted classification), return True """
def attack_success(cleancrop,x, initial_pic, target_class, searchspace, \
    sticker,opstickercv,magnification, zstore, facemask, targeted_attack=False):
    """
    determine whether the current optimal solution can successfully attack
    """
    attack_image, valid = predict.perturb_image(x, initial_pic, \
        sticker, opstickercv, magnification, zstore, searchspace, facemask)

    if(threat_model == 'facenet'):
        rank, _ = predict.predict_type_facenet(attack_image,cleancrop)
    elif(threat_model == 'arcface'):
        rank, _ = predict.predict_type_arcface(attack_image,cleancrop)
    elif(threat_model == 'sphereface'):
        rank, _ = predict.predict_type_sphereface(attack_image,cleancrop)
    elif(threat_model == 'cosface'):
        rank, _ = predict.predict_type_cosface(attack_image,cleancrop)
    elif(threat_model == 'yolov5'):
        rank, _ = predict.predict_type_yolov5(attack_image, cleancrop)
    elif(threat_model == 'taolipai'):
        rank, _ = predict.predict_type_taolipai(attack_image,target_class,cleancrop)
    elif(threat_model == 'retrieval'):
        rank, _ = predict.predict_type_retrieval(attack_image,target_class, cleancrop)
# predicted_class = rank[0][0]
    predicted_class = rank[0]
    #print('callback: predicted_class=',predicted_class,'valid[0]=',valid[0],x)
    if ((targeted_attack and predicted_class == target_class and valid[0]==1) or
        (not targeted_attack and predicted_class != target_class and valid[0]==1)):
        return True
    # NOTE: return None otherwise (not False), due to how Scipy handles its callback function

def num_clip(l,u,x):
    if(x>=l):
        if(x<=u):
            r = x
        else:
            r = u
    else:
        r = l
    return r

def region_produce(cleancrop,xs, true_label, searchspace, pack_searchspace, trace_searchspace, initial_pic, \
    sticker,opstickercv,magnification, zstore, facemask):
    """
    Inbreeding strategy
    """
    h, w = int(facemask.shape[0]), int(facemask.shape[1])
    len_relative = len(xs)
    len_per = np.zeros((len_relative,1))  # the number of valid dots around the current dot
    pots = []                             # The whole set of perturbation vectors considered in inbreeding
    inbreeding = []
    for i in list(range(len_relative)):
        cur = int(xs[i][0])
        alp = xs[i][1]
        angle = xs[i][2]
        x = int(searchspace[cur][0])
        y = int(searchspace[cur][1])
        neighbors = adjacent_coordinates(x,y,s=1)
        temp = 0
        for j in range(len(neighbors)):
            p = num_clip(0,w-1,int(neighbors[j][0]))
            q = num_clip(0,h-1,int(neighbors[j][1]))
            if(alp in trace_searchspace[q][p]):
                judge = random.random()
                if(judge <= 0.5):                            # change the step
                    slide = 2
                    while(1):
                        #print('change step')
                        far_neighbors = adjacent_coordinates(x,y,s=slide)
                        pn = int(far_neighbors[j][0])
                        qn = int(far_neighbors[j][1])
                        if(alp in trace_searchspace[qn][pn]):
                            slide = slide + 1
                        else:
                            break
                    trace_searchspace[qn][pn].append(alp)
                    attribute = pack_searchspace[qn][pn]
                    if(attribute >= 0):
                        temp = temp + 1
                        pots.append([attribute,alp,angle])
                else:                                        # change the alpha using random
                    #print('change alpha')
                    attribute = pack_searchspace[q][p]
                    alp_ex = random.uniform(0.8,0.98)
                    if(attribute >= 0):
                        temp = temp + 1
                        pots.append([attribute,alp_ex,angle])
                    trace_searchspace[q][p].append(alp_ex)
            else:
                trace_searchspace[q][p].append(alp)
                attribute = pack_searchspace[q][p]
                #print('attribute = ',attribute)
                if(attribute >= 0):
                    temp = temp + 1
                    pots.append([attribute,alp,angle])
        len_per[i][0] = temp
    predictions = single_predict(cleancrop,pots, initial_pic, true_label, searchspace, \
        sticker,opstickercv,magnification, zstore, facemask)
    cursor = 0
    #print('len_per = ',len_per.T)
    for i in range(len_relative):
        sublen = len_per[i][0]
        if(sublen != 0):
            upper = int(cursor + sublen)
            subset = predictions[int(cursor):upper]
            better = np.argsort(subset)[0]
            inbreeding.append(pots[int(cursor+better)])
        else:
            inbreeding.append(xs[i])
        cursor = cursor + sublen
    
    #print('len_relative, inbreeding = ',len_relative, len(inbreeding))
    return inbreeding


def adjacent_coordinates(x,y,s):
    adj = []
    adj.append([x-s,y-s])
    adj.append([x,y-s])
    adj.append([x+s,y-s])
    adj.append([x-s,y])
    adj.append([x+s,y])
    adj.append([x-s,y+s])
    adj.append([x,y+s])
    adj.append([x+s,y+s])
    return adj


def attack(idx,true_label,initial_pic,sticker,opstickercv,magnification,\
    cleancrop,zstore,target=None, maxiter=3, popsize=2):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else true_label

    # facemask = feature.make_mask(initial_pic) # valid=1, unvalid=0
    # mask_path = '/home/ubuntu/Documents/RY2020/patch/facemask4/{}.jpg'.format(idx)
    # facemask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)/255
    
    # facemask = feature.make_mask_yolov5(initial_pic, cleancrop)
    # facemask = feature.make_mask_taolipai(initial_pic, cleancrop)
    facemask = feature.make_mask_retrieval(initial_pic, cleancrop)
    

    num_space = np.sum(facemask).astype(int)
    searchspace = np.zeros((num_space,2)) # store the coordinate(Image style)
    pack_searchspace = copy.deepcopy(facemask)-2  # valid=-1, unvalid=-2
    trace_searchspace = []
    for i in range(facemask.shape[0]):
        col = [[-1] for j in range(facemask.shape[1])]
        trace_searchspace.append(col)
    k = 0
    for i in range(facemask.shape[0]):
        for j in range(facemask.shape[1]):
            if(facemask[i][j] == 1): 
                searchspace[k] = (j,i)   # facemask.shape[0] -> height, facemask.shape[1] -> width
                k = k + 1
    np.random.shuffle(searchspace)
    for i in range(len(searchspace)):
        x = int(searchspace[i][0])
        y = int(searchspace[i][1])
        pack_searchspace[y][x] = int(i)
    bounds = [(0,num_space), (0.8,0.98),(0,359)]
    print('---------begin attack---------------')
    
    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs,gorb):
        return predict_classes(cleancrop,xs, gorb, initial_pic, target_class, searchspace, \
            sticker,opstickercv,magnification, zstore, facemask, target is None)
    
    def callback_fn(x, convergence):
        return attack_success(cleancrop,x, initial_pic, target_class, searchspace, \
            sticker,opstickercv,magnification, zstore, facemask, targeted_attack)
    
    def region_fn(xs):
        return region_produce(cleancrop,xs, true_label, searchspace, pack_searchspace, trace_searchspace, \
            initial_pic, sticker,opstickercv,magnification, zstore, facemask)
    
    def ct_energy(ranks, pred_ps, valids):
        return convert_energy(ranks, pred_ps, valids, target_class)
    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, region_fn, ct_energy, bounds, maxiter=maxiter, popsize=popsize,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image, valid = predict.perturb_image(attack_result.x, initial_pic, \
        sticker, opstickercv, magnification, zstore, searchspace, facemask)
    
    # if(threat_model == 'facenet'):
    #     rank, pred_p = predict.predict_type_facenet([initial_pic],cleancrop)
    #     rank2, pred_p2 = predict.predict_type_facenet(attack_image,cleancrop)
    # elif(threat_model == 'arcface'):
    #     rank, pred_p = predict.predict_type_arcface([initial_pic],cleancrop)
    #     rank2, pred_p2 = predict.predict_type_arcface(attack_image,cleancrop)
    # elif(threat_model == 'sphereface'):
    #     rank, pred_p = predict.predict_type_sphereface([initial_pic],cleancrop)
    #     rank2, pred_p2 = predict.predict_type_sphereface(attack_image,cleancrop)
    # elif(threat_model == 'cosface'):
    #     rank, pred_p = predict.predict_type_cosface([initial_pic],cleancrop)
    #     rank2, pred_p2 = predict.predict_type_cosface(attack_image,cleancrop)
    # elif(threat_model == 'yolov5'):
    #     rank, pred_p = predict.predict_type_yolov5([initial_pic], cleancrop)
    #     rank2, pred_p2 = predict.predict_type_yolov5(attack_image, cleancrop)
    # elif(threat_model == 'taolipai'):
    #     rank, pred_p = predict.predict_type_taolipai([initial_pic],target_class, cleancrop)
    #     rank2, pred_p2 = predict.predict_type_taolipai(attack_image,target_class, cleancrop)
    # elif(threat_model == 'retrieval'):
    #     rank, pred_p = predict.predict_type_retrieval([initial_pic],target_class, cleancrop)
    #     rank2, pred_p = predict.predict_type_retrieval(attack_image,target_class, cleancrop)
    #attack_image[0].save('/home/guoying/patch/trans_rhde/check_sph_unt_single/{}_{}.jpg'.format(sticker_name,idx))
    #attack_image[0].save('/home/guoying/patch/trans_rhde/physical_imgs/{}_{}_{}.jpg'.format(threat_model,sticker_name,idx))
    '''
    prior_probs = pred_p[0][target_class].item()
    predicted_class = rank2[0][0]
    predicted_probs = pred_p2[0][predicted_class].item()
    d1 = [rank[0][0],rank[0][1]]
    score1 = [pred_p[0][rank[0][0]].item(),pred_p[0][rank[0][1]].item()]
    d2 = [rank2[0][0],rank2[0][1]]
    score2 = [pred_p2[0][rank2[0][0]].item(),pred_p2[0][rank2[0][1]].item()]
    '''

    # prior_probs = pred_p[0]
    # predicted_class = rank2[0]
    # predicted_probs = pred_p2[0]
    # d1 = [rank[0]]
    # score1 = [pred_p[0]]
    # d2 = [rank2[0]]
    # score2 = [pred_p2[0]]

    # actual_class = true_label
    # success = (predicted_class != actual_class ) and valid[0]==1
    # # cdiff = pred_p[0][actual_class].item() - pred_p2[0][actual_class].item()
    # cdiff = pred_p[0] - pred_p2[0]

    # sid = int(attack_result.x[0])
    # x = int(searchspace[sid][0])
    # y = int(searchspace[sid][1])
    # factor = attack_result.x[1]
    # angle = attack_result.x[2]
    # vector = [x, y, factor, angle, sid, attack_result.x[0]]

    # return [actual_class, predicted_class, success, cdiff, prior_probs, predicted_probs, vector,d1,score1,d2,score2]

def getScoreList():
    scoreList = []
    K = 5
    for i in range(K):
        scoreList.append((np.math.pow(2,K - i)-1) / (np.math.pow(2,K) - 1))    
    return scoreList

def getBBox():
    mp = {}
    # fr = open("/content/gdrive/My Drive/Category and Attribute Prediction Benchmark/Anno/list_bbox.txt","r")
    fr = open("E:/In-shop Clothes Retrieval Benchmark/Anno/list_bbox.txt","r")
    fr.readline()
    fr.readline()
    for line in fr.readlines():
        line = line.strip()
        strs = line.split(" ")
        blist = []
        blist.append(int(strs[-4]))
        blist.append(int(strs[-3]))
        blist.append(int(strs[-2]))
        blist.append(int(strs[-1]))
        mp[strs[0]] = blist
    return mp


if __name__=="__main__":
    model_set = ['arcface', 'facenet', 'sphereface','cosface', 'yolov5', 'taolipai', 'retrieval']
    bound = 15             # critical value of the gap between label1 and label2
    threat_model = model_set[-1]
    tempnum = ''
    pinf, ninf = 99.9999999, 0.0000001
    convert = False          # indicate whether DE needs to re-compute energies to compare with target result
    start = False            # whether start target attack from untarget style
    latter = 0               # target class

    data_dir = './data'
    dataset = datasets.ImageFolder(data_dir)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}

    #with open('/home/guoying/patch/celeba_align/celeba_sph_obj.txt') as f:
        #inds = [int(line[:-1].strip()) for line in f.readlines()]

    # inds = [0,1,2]  # picture list (total two pictures)
    #dir = 'E:/In-shop Clothes Retrieval Benchmark/attack/'
# /content/gdrive/My Drive/Category and Attribute Prediction Benchmark/output3/attackImg/
    # dir = "/content/gdrive/My Drive/Category and Attribute Prediction Benchmark/attack/"
    # picList_ = os.listdir(dir)
    # picList = []
    # for item in picList_:
    #     picList.append("attack/"+item)
    # print(picList)

    bboxMp = getBBox()
    
    # picList = ["img/Sheer_Pleated-Front_Blouse/img_00000005.jpg","img/Sheer_Pleated-Front_Blouse/img_00000060.jpg","img/Los_Angeles_Graphic_Tank/img_00000025.jpg","img/Los_Angeles_Graphic_Tank/img_00000003.jpg","img/Los_Angeles_Muscle_Tee/img_00000014.jpg","img/Los_Angeles_Muscle_Tee/img_00000023.jpg","img/Lost_&_Found_Wool-Blend_Sweater/img_00000041.jpg","img/Lost_&_Found_Wool-Blend_Sweater/img_00000012.jpg","img/Lost_&_Found_Wool-Blend_Sweater/img_00000016.jpg","img/Lost_&_Found_Wool-Blend_Sweater/img_00000014.jpg","img/Performance_Shorts/img_00000008.jpg","img/Performance_Shorts/img_00000009.jpg","img/Performance_Shorts/img_00000064.jpg","img/Paisley_Pattern_Pencil_Skirt/img_00000039.jpg","img/Paisley_Pattern_Pencil_Skirt/img_00000010.jpg","img/Paisley_Pattern_Pencil_Skirt/img_00000016.jpg","img/Y-Back_Halter_Dress/img_00000006.jpg","img/Y-Back_Halter_Dress/img_00000028.jpg","img/Y-Back_Halter_Dress/img_00000044.jpg"]
    # picList = ["img/Los_Angeles_Graphic_Tank/img_00000025.jpg","img/Los_Angeles_Graphic_Tank/img_00000003.jpg","img/Los_Angeles_Muscle_Tee/img_00000014.jpg","img/Los_Angeles_Muscle_Tee/img_00000023.jpg","img/Lost_&_Found_Wool-Blend_Sweater/img_00000041.jpg","img/Lost_&_Found_Wool-Blend_Sweater/img_00000012.jpg","img/Lost_&_Found_Wool-Blend_Sweater/img_00000016.jpg","img/Lost_&_Found_Wool-Blend_Sweater/img_00000014.jpg","img/Performance_Shorts/img_00000008.jpg","img/Performance_Shorts/img_00000009.jpg","img/Performance_Shorts/img_00000064.jpg","img/Paisley_Pattern_Pencil_Skirt/img_00000039.jpg","img/Paisley_Pattern_Pencil_Skirt/img_00000010.jpg","img/Paisley_Pattern_Pencil_Skirt/img_00000016.jpg","img/Y-Back_Halter_Dress/img_00000006.jpg","img/Y-Back_Halter_Dress/img_00000028.jpg","img/Y-Back_Halter_Dress/img_00000044.jpg"]
    # picList = ["img/Contrast_Trim_Henley/img_00000002.jpg","img/Contrast_Trim_Henley/img_00000031.jpg","img/Contrast_Trim_Henley/img_00000043.jpg","img/Contrast_Trim_Henley/img_00000046.jpg","img/Contrast-Front_Knit_Sweater/img_00000002.jpg","img/Contrast-Front_Knit_Sweater/img_00000004.jpg","img/Contrast-Front_Knit_Sweater/img_00000010.jpg","img/Contrast-Front_Knit_Sweater/img_00000028.jpg","img/Contrast-Front_Knit_Sweater/img_00000025.jpg","img/Corduroy_Hooded_Parka/img_00000006.jpg","img/Corduroy_Hooded_Parka/img_00000011.jpg","img/Corduroy_Hooded_Parka/img_00000058.jpg","img/Corduroy_Hooded_Parka/img_00000077.jpg","img/Corduroy_Hooded_Parka/img_00000076.jpg","img/Abstract_Print_Woven_Cardigan/img_00000004.jpg","img/Abstract_Print_Woven_Cardigan/img_00000033.jpg"]
    # for i in range(10):
    #     picList.append("img/Cowl_Neck_Sweater/img_0000000{}.jpg".format(i))
    # for i in range(10,50):
    #     picList.append("img/Cowl_Neck_Sweater/img_000000{}.jpg".format(i))
    # picList = ["img/Cowl_Neck_Sweater/img_00000056.jpg"]

    sticker_name = 'complex'
    stickerpath = 'C:/Users/admin/Desktop/codes_rhde/sticker/{}.png'.format(sticker_name)
    # stickerpath = '/content/gdrive/My Drive/codes_rhde/sticker/{}.png'.format(sticker_name)
    stickerpic = Image.open(stickerpath)
    scale1 = stickerpic.size[0]//23
    scale2 = 12
    # scale1 = stickerpic.size[0]//46
    # scale2 = 24
    magnification = scale2/scale1
    operate_sticker = stick.change_sticker(stickerpic,scale1)
    sticker = stick.change_sticker(stickerpic,scale2)
    opstickercv = rotate.img_to_cv(operate_sticker)

    generate_rank, generate_score, best_rank, best_score = [],[],[],[]
    timess = 0  # Record the query times to visit the network when each image generates adversarial example
    hh = 0

    picList = []
    nameList = os.listdir(r"E:\In-shop Clothes Retrieval Benchmark\attack")
    for name in nameList:
        picList.append("attack/"+name)
    # for i in range(len(inds)):
    fr = open("E:/In-shop Clothes Retrieval Benchmark/pic_list.txt")
    basepicList = []
    for line in fr.readlines():
        line = line.strip()
        basepicList.append(line)
    for initial_pic,base_pic,i in zip(picList,basepicList,range(len(picList))):
        predict.attackRoundID = i
        predict.aq = 0
        predict.finishFlag = False
        # print('i = ',i,'idx = ',inds[i])
        # idx = int(inds[i])
        # initial_pic = dataset[idx][0]
        # true_label = dataset[idx][1]
        # initial_pic = dataset.imgs[idx][0]
        # initial_pic = 'img/Asymmetrical_Hem_Top/img_00000010.jpg'
        # initial_pic = 'img/Sheer_Pleated-Front_Blouse/img_00000005.jpg'
        # initial_pic = r'img\Striped_Slub_Knit_Tank\img_00000015.jpg'
        # initial_pic = 'img/MEN/Sweaters/id_00000702/04_1_front.jpg'
        true_label = dataset.imgs[0][1]
        target_class = dataset.idx_to_class[true_label]
        if(threat_model == 'facenet'):
            rank, _, cleancrop = predict.initial_predict_facenet([initial_pic])
        elif(threat_model == 'arcface'):
            rank, _, cleancrop = predict.initial_predict_arcface([initial_pic])
        elif(threat_model == 'sphereface'):
            rank, _, cleancrop = predict.initial_predict_sphereface([initial_pic])
        elif(threat_model == 'cosface'):
            rank, _, cleancrop = predict.initial_predict_cosface([initial_pic])
        elif(threat_model == 'yolov5'):
            rank, _, cleancrop = predict.initial_predict_yolov5([initial_pic])
        elif(threat_model == 'taolipai'):
            rank, _, cleancrop = predict.initial_predict_taolipai([initial_pic],dataset.idx_to_class[true_label])
        elif(threat_model == 'retrieval'):
            predict.true_labels = getLabelLists(initial_pic)
            predict.scoreList = getScoreList()
            rank = [dataset.idx_to_class[true_label]]
            cleancrop = bboxMp[base_pic]
            predict.orginalRetrieval = predict.initial_predict_retrieval(initial_pic)
            # rank, _, cleancrop = predict.initial_predict_retrieval(initial_pic,dataset.idx_to_class[true_label])
        #print(rank[0][0])

        # if(rank[0][0] != true_label):
            # continue
        if(dataset.idx_to_class[true_label] != rank[0]):
            continue
        else:
            t1 = time.time()
            W, H = dataset[0][0].size
            zstore = np.zeros((H, W, 2))
            # zstore = mapping3d.generate_zstore(initial_pic)
            # r = attack(idx,true_label, initial_pic, sticker,opstickercv,magnification,cleancrop,zstore)
            attack(i,dataset.idx_to_class[true_label], initial_pic, sticker,opstickercv,magnification,cleancrop,zstore)
            t2 = time.time()
            
            print('idx = ',i,"one picture's time is {}".format(t2-t1))
            
            timess = 0
            convert = False  # indicate whether DE needs to re-compute energies to compare with target result
            start = False    # whether start target attack from untarget style
            latter = 0       # target class
            generate_rank, generate_score, best_rank, best_score = [],[],[],[]
            hh = hh + 1
        
