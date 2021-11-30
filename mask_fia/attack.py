from heuristicsDE import differential_evolution
import numpy as np
from torchvision import transforms 
import torch

trans = transforms.Compose([
                transforms.ToTensor(),
            ])
MAX = 10000
inf = -MAX
suf = MAX
CLUSTER = 10   # mask中每cluster个相邻的位置用一个数表示
WIDTH=224
HEIGHT=224

def trans2mask(xs, width=224, height=224):
    # param xs:种群所有个体，每个个体是一个用数字构成的mask
    # param width:模型输入的宽
    # param height:
    msks = None
    for each in xs:#每一个mask
        l = []
        for n,each2 in enumerate(each):#每一个数
            v = str(bin(each2))
            l = l + [int(b) for b in v[2:]]
        l = np.array(l)
        msk = trans(l.resize((height,width,1)))
        if msks == None:
            msks = msk
        else:
            msks = torch.cat((msks,msk),0)
    # return msks tensor([num,height,width])
    return msks

def trans2inbreedings(masks):
    # inbreeding:由0-1023的数组成的列表
    inbreedings = []
    for mask in masks:
        inbreeding=[]
        num_str = ''
        for i in range(len(mask[0])):
            for j in range(len(mask[1])):
                if len(num_str)==CLUSTER :
                    inbreeding.append(int(num_str,2))
                    num_str=''
                num_str+=str(mask[i][j])
        if len(num_str)!=0:
            inbreeding.append(int(num_str, 2))
        inbreedings.append(inbreeding)
    return inbreedings

def predict_transfer_score(mask):
    # 每个个体的得分，通过每个mask单独作用到图像进行对抗攻击产生对抗样本在一组黑盒模型上的效果得分获得
    score=None  # TODO
    return score

def predict_n_transfer_score(masks):
    # 每个个体的得分，通过每个mask单独作用到图像进行对抗攻击产生对抗样本在一组黑盒模型上的效果得分获得
    scores = []
    for mask in masks:
        score=predict_transfer_score(mask)
        scores.append(score)
    return scores

def predict_classes(xs, bounds):
    masks = trans2mask(xs,WIDTH,HEIGHT)
    return predict_n_transfer_score(masks)

def attack_success():
    return True

def mask_extent(mask):
    # 对杂交mask进行四次扩展(上下,左右,上下左右,一圈),
    # return: 4个扩展后的mask组成的masks
    masks=[]
    # 上下扩展
    mask_temp = np.zeros((HEIGHT, WIDTH))
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if mask[i][j]==1:
                mask_temp[i][j] = 1
                if not i==0 and not i==HEIGHT-1:
                    mask_temp[i - 1][j] = 1
                    mask_temp[i + 1][j] = 1
                elif i==0:
                    mask_temp[i + 1][j] = 1
                else:
                    mask_temp[i - 1][j] = 1
    masks.append(mask_temp)
    # 左右扩展
    mask_temp = np.zeros((HEIGHT, WIDTH))
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if mask[i][j] == 1:
                mask_temp[i][j] = 1
                if not j == 0 and not j == WIDTH - 1:
                    mask_temp[i][j-1] = 1
                    mask_temp[i][j+1] = 1
                elif j == 0:
                    mask_temp[i][j+1] = 1
                else:
                    mask_temp[i][j-1] = 1
    masks.append(mask_temp)
    # 上下左右扩展
    mask_temp = np.zeros((HEIGHT, WIDTH))
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if mask[i][j] == 1:
                mask_temp[i][j] = 1
                if not j == 0 and not j == WIDTH - 1:
                    mask_temp[i][j - 1] = 1
                    mask_temp[i][j + 1] = 1
                elif j == 0:
                    mask_temp[i][j + 1] = 1
                else:
                    mask_temp[i][j - 1] = 1

                if not i==0 and not i==HEIGHT-1:
                    mask_temp[i - 1][j] = 1
                    mask_temp[i + 1][j] = 1
                elif i==0:
                    mask_temp[i + 1][j] = 1
                else:
                    mask_temp[i - 1][j] = 1
    masks.append(mask_temp)

    # 一圈扩展
    mask_temp = np.zeros((HEIGHT, WIDTH))
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if mask[i][j] == 1:
                mask_temp[i][j] = 1
                if not j == 0 and not j == WIDTH - 1:
                    mask_temp[i][j - 1] = 1
                    mask_temp[i][j + 1] = 1
                elif j == 0:
                    mask_temp[i][j + 1] = 1
                else:
                    mask_temp[i][j - 1] = 1

                if not i == 0 and not i == HEIGHT - 1:
                    mask_temp[i - 1][j] = 1
                    mask_temp[i + 1][j] = 1
                elif i == 0:
                    mask_temp[i + 1][j] = 1
                else:
                    mask_temp[i - 1][j] = 1

                if not i==0 and not j==0:
                    mask_temp[i - 1][j - 1] = 1
                if not i==HEIGHT-1 and not j== 0:
                    mask_temp[i + 1][j - 1] = 1

                if not i == 0 and not j == WIDTH-1:
                    mask_temp[i - 1][j + 1] = 1

                if not i == HEIGHT and not j == WIDTH-1:
                    mask_temp[i + 1][j + 1] = 1
    masks.append(mask_temp)
    return masks
def bestdir_of_extent(mask):

    masks=mask_extent(mask)
    socres=predict_n_transfer_score(masks)
    extented_best_mask=masks[np.argmin(socres)]
    return extented_best_mask

def region_produce(xs, WIDTH,HEIGHT,alpha = 0.1):
    # 产生uP的后代，产生后代策略
    # param xs:
    # param alpha: 两个mask的交集的比例大于alpha时认为共性高
    
    masks = trans2mask(xs,WIDTH,HEIGHT)
    mskn = masks.shape[0]
    sim_table = np.zeros((mskn,mskn))
    for i in range(len(mskn)):
        for j in range(len(mskn)):
            if i == j:  sim_table[i][j] = suf
            elif i > j: sim_table[i][j] = sim_table[j][i]
            else:
                ni = masks[i].sum()
                nj = masks[j].sum()
                nij = (masks[i]*masks[j]).sum()
                if nij < alpha*(ni+nj)/2:
                    sim_table[i][j] = inf
                else:
                    sim_table[i][j] = nij
    # 找每个节点（mask）相似的节点集，节点集并上该节点不能存在无向环
    set1 = []
    set2 = []
    for i in range(len(mskn)):
        cur_set = []
        nodes = []
        weights = []
        for j in range(len(mskn)):
            if i!=j and sim_table[i][j]!=inf:
                nodes.append(j)
                weights.append(sim_table[i][j])
        if len(nodes) == 0:#独立mask评分高且与所有mask交集少
            set2.append(i)
        else:
            weights = np.array(weights)
            inds = np.argsort(-weights)
            nodes = nodes[inds]
            cur_set.append(nodes[0])
            for j in nodes[1:]:
                flag = True
                for k in cur_set:
                    if sim_table[j][k] != inf:
                        if (j,i) in set1:
                            set1.remove((j,i))
                        flag = False
                        break
                if flag:
                    cur_set.append(j)
            for j in cur_set:
                set1.append((i,j))
    set3 = []
    set3 = [elem for elem in set1 if (elem[1], elem[0]) not in set3]
    set1 = set3
    set3 = []
    if len(set1) + len(set2) > mskn:
        # 去掉杂交产生的多余个体
        scorelist = [sim_table[elem[0]][elem[1]] for elem in set1]
        scorelist = np.array(scorelist)
        indices = np.argsort(scorelist)
        set1 = set1[indices]
        set1 = set1[:mskn - len(set2)]
    elif len(set1) + len(set2) < mskn:
        # 补齐杂交少产生的个体
        full_set = set(list(range(mskn)))
        extra_set = set(set2)
        set3 = full_set - extra_set
        set3 = list(set3)
        set3 = set3[:mskn-len(set1)-len(set2)]

    inbreeding_masks = []
    # set1杂交，扩展
    for (i,j) in set1:
        mask = masks[i]*masks[j]
        mask = bestdir_of_extent(mask)
        inbreeding_masks.append(mask)
    # 将msk转为个体
    return trans2inbreedings(inbreeding_masks),set2,set3

def attack(maxiter = 40, popsize = 20, width = 224, height = 224):

    bounds = [(0, 1048575)] * (int(width*height/CLUSTER)) + [0, int(np.math.pow(2, width*height%CLUSTER))-1]
    print('---------begin attack---------------')
    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return predict_classes(xs)
    
    def callback_fn(x, convergence):
        return attack_success()
    
    def region_fn(xs):
        return region_produce(xs)

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, region_fn, bounds, maxiter=maxiter, popsize=popsize,
        recombination=1, atol=-1, callback=callback_fn, polish=False)




if __name__=="__main__":
    model_set = ['arcface', 'facenet', 'sphereface','cosface', 'yolov5', 'taolipai', 'retrieval']
    threat_model = model_set[-1]
    maxiter = 40
    popsize = 20
    width = 224
    height = 224
    attack(maxiter, popsize, width, height)