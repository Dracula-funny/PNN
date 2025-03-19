#Adapted from Swintransformer
import collections
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def protoPred(output, target, topk=(1,)):

    maxk = min(max(topk), output.size()[1])

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    return pred



def addImages(proto_pred,knn_pred,indices,knn_train_lable):
    proto_pred = proto_pred[0]
    label_diff = torch.abs(knn_train_lable[indices] - knn_pred.unsqueeze(1).expand(-1, indices.shape[1]))
    zero_count = torch.sum(label_diff == 0, dim=1)
    addImage = []
    addLabel = []
    for i,j in enumerate(proto_pred):
        if (proto_pred[i]==knn_pred[i] and zero_count[i]>3):
            addImage.append(i)
            addLabel.append(j)

    return addImage,addLabel


def calculate_accuracy(predictions, true_labels):

    assert len(predictions) == len(true_labels), "error"

    correct_predictions = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct_predictions / len(true_labels)

    return accuracy


def addImages2(support_features,support_label,query_features,labelnum,temp):
    if support_features.dim() == 4:
        support_images = F.adaptive_avg_pool2d(support_features, 1).squeeze_(-1).squeeze_(-1)
        query_images = F.adaptive_avg_pool2d(query_features, 1).squeeze_(-1).squeeze_(-1)

    assert support_images.dim() == query_images.dim() == 2
    support_images = F.normalize(support_images, p=2, dim=1, eps=1e-12)
    query_images = F.normalize(query_images, p=2, dim=1, eps=1e-12)

    knn_support_images = torch.cat((support_images, query_images), dim=0)

    nearest_neighbors = NearestNeighbors(n_neighbors=4, metric='cosine').fit(knn_support_images.cpu().detach().numpy())

    distances, indices = nearest_neighbors.kneighbors(support_images.cpu().detach().numpy())

    query_label = [-1 for _ in range(labelnum)]

    indices2 = indices[:, 1:]

    for i ,j in enumerate(indices2):

        for j1 in j:

            if j1>=temp:
                j2 = j1-temp
                if(query_label[j2]<-1):
                    continue
                elif(query_label[j2]>-1 and query_label[j2]!=support_label[i]):
                    query_label[j2]=-2
                elif (query_label[j2] > -1 and query_label[j2] == support_label[i]):
                    continue
                else:
                    query_label[j2]=support_label[i].item()

    return query_label

def isThan16(label):

    num1 = 0


    count_list = [0] * 5

    for num in label:
        if num>-1 and num<5:
            count_list[num] += 1
        if num>4:
            print(label)


    for j1, j2 in enumerate(count_list):
        if j2 > 16:
            num1 += 1

    return num1


def stastic_balance(idx1,idx2,real_idx):

    sum = len(idx1)
    idx1_acc = torch.sum(idx1 == real_idx)
    idx2_acc = torch.sum(idx2 == real_idx)

    return idx1_acc,idx2_acc,sum

def class_balance(proto_score,knn_indices,knn_trainLabel,queryImages_idx,queryLabel,k):

    temp_num = 16

    images_idx = queryImages_idx
    label = queryLabel
    updata_idx = []

    count_list = [0] * 5

    for num in label:
        count_list[num] += 1

    greater_than_5 = 0
    greater_than_5_idx = []
    for j1,j2 in enumerate(count_list) :
        if j2>temp_num:
            greater_than_5+=1
            greater_than_5_idx.append(j1)

    has_greater_than_16 = any(count > temp_num for count in count_list)
    proto_confi = torch.zeros(75, 5)
    knnConfi = torch.zeros(75, 5)

    if has_greater_than_16:


        logits2 = torch.exp(proto_score)
        sum_logits = torch.sum(logits2, dim=1)

        proto_confi = logits2/sum_logits.reshape(75,1)


        numLabel0 = torch.sum(knn_trainLabel[knn_indices] == 0, dim=1)
        numLabel1 = torch.sum(knn_trainLabel[knn_indices] == 1, dim=1)
        numLabel2 = torch.sum(knn_trainLabel[knn_indices] == 2, dim=1)
        numLabel3 = torch.sum(knn_trainLabel[knn_indices] == 3, dim=1)
        numLabel4 = torch.sum(knn_trainLabel[knn_indices] == 4, dim=1)

        knnConfi0 = numLabel0 / k
        knnConfi1 = numLabel1 / k
        knnConfi2 = numLabel2 / k
        knnConfi3 = numLabel3 / k
        knnConfi4 = numLabel4 / k

        knnConfi0 = knnConfi0.reshape(75,1)
        knnConfi1 = knnConfi1.reshape(75, 1)
        knnConfi2 = knnConfi2.reshape(75, 1)
        knnConfi3 = knnConfi3.reshape(75, 1)
        knnConfi4 = knnConfi4.reshape(75, 1)
        knnConfi = torch.cat((knnConfi0, knnConfi1, knnConfi2, knnConfi3, knnConfi4), dim=1)

        confi = proto_confi+0.1*knnConfi



        label_self_index = [-1]*75

        for labelidx1,labelidx2 in enumerate(images_idx):
            label_self_index[labelidx2] = labelidx1

        for than_5_idx in greater_than_5_idx:
            confi_temp = {}
            for images_idx1,images_idx2 in enumerate(images_idx):
                if label[images_idx1] ==than_5_idx:
                    confi_temp[images_idx2] = confi[images_idx2][than_5_idx].item()

            sorted_dict = dict(sorted(confi_temp.items(), key=lambda item: item[1], reverse=True))

            keys = list(sorted_dict.keys())

            add_images_idx = []
            for keys_idx in range(len(keys)):
                if keys_idx>temp_num-1:
                    add_images_idx.append(keys[keys_idx])
                    updata_idx.append(keys[keys_idx])

            for x in add_images_idx:
                confi_line = confi[x]
                sorted_indices = sorted(range(len(confi_line)), key=lambda x: confi_line[x], reverse=True)

                for x1 in sorted_indices:
                    if(count_list[x1]<temp_num):

                        label[label_self_index[x]] = x1

                        count_list[than_5_idx]-=1
                        count_list[x1]+=1
                        break

    count_list2 = [0] * 5


    for num in label:
        count_list2[num] += 1


    greater_than_51 = 0

    for j1, j2 in enumerate(count_list2):
        if j2 > temp_num:
            greater_than_51 += 1

    return images_idx,label,updata_idx,proto_confi,knnConfi


def getprotoconfi(logits2,label):
    logits2 = torch.exp(logits2)
    sum_logits = torch.sum(logits2, dim=1)


    selected_logits = logits2[range(logits2.size(0)), label]
    confi1 = selected_logits / sum_logits

    confisum2 = torch.sum(confi1)
    newconfi2 = confi1 / confisum2


    return confi1.cuda(),newconfi2.cuda()




def knn_st(knn_distances,knn_test_lable,indices,x1,x2,k):

    disSum = torch.zeros(x1, x2)

    for i in range(x1):
        for j in range(k):
            label = (knn_test_lable[indices])[i, j]
            distance = knn_distances[i, j]
            disSum[i, label] += distance


    count_tensor = torch.zeros(x1, x2, dtype=torch.int, device='cuda')
    for i in range(x2):
        count_tensor[:, i] = (knn_test_lable[indices] == i).sum(dim=1)


    max_indices = []
    for i in range(x1):
        row = count_tensor[i]
        max_value = torch.max(row).item()
        indices2 = (row == max_value).nonzero()
        max_indices.append(indices2)


    element_counts = []

    for tensor in max_indices:

        num_elements = tensor.size(0)
        element_counts.append(num_elements)


    res = torch.empty(0, dtype=torch.int, device='cuda')



    num=0
    for i, j in enumerate(element_counts, 0):
        if(j>1):
            #max_indices[i]
            min_values, min_indices = torch.max(disSum[i][max_indices[i]], dim=0)

            res = torch.cat((res, max_indices[i][min_indices].to(torch.int)))
            num+=1
        else:
            res = torch.cat((res, max_indices[i].to(torch.int)))

    return res.t()[0]




def getknnconfi(indices,knn_train_lable,knn_test_lable,k):

    label_diff = torch.abs(knn_train_lable[indices] - knn_test_lable.unsqueeze(1).expand(-1, indices.shape[1]))

    non_zero_count = torch.sum(label_diff == 0, dim=1)
    newconfi = (non_zero_count.float() / k)
    confisum2 = newconfi.sum()

    newconfi2 = newconfi/confisum2

    return newconfi.cuda(),newconfi2.cuda()



def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, topK, step):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config,
                  'step':step}
    if topK is not None:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}_top{topK}.pth')
    else:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def delete_checkpoint(config, topK=None, epoch = None):
    if topK is not None:
        for file_ in os.listdir(config.OUTPUT):

            if f"top{config.SAVE_TOP_K_MODEL}" in file_:
                os.remove(os.path.join(config.OUTPUT, file_))
                break
        for j in range(config.SAVE_TOP_K_MODEL-1,topK-1, -1):

            for file_ in os.listdir(config.OUTPUT):
                if f"top{j}" in file_:
                    os.rename(os.path.join(config.OUTPUT, file_),
                        os.path.join(config.OUTPUT, file_).replace(f"top{j}", f"top{j+1}"))
                    break
    elif epoch is not None:
        if os.path.exists(os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")):
            os.remove(os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth"))
        
def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

    possible_keys = ["state_dict", "model", "models", "state","params"]

    flag = True
    for key in possible_keys:
        if key in checkpoint.keys():
            the_key = key
            flag = False
            break
    if flag:
        state_dict = checkpoint
    else:
        state_dict = checkpoint[the_key]

    state_keys = list(state_dict.keys())
    for i, key in enumerate(state_keys):
        if "backbone" in key:
            newkey = key.replace("backbone.", "")
            state_dict[newkey] = state_dict.pop(key)
        if "module.feature" in key:
            newkey = key.replace("module.feature.", "")
            state_dict[newkey] = state_dict.pop(key)
        if "module.l" in key:
            newkey = key.replace("module.", "")
            state_dict[newkey] = state_dict.pop(key)
        if "encoder.layer1.0." in key:
            newkey = key.replace("encoder.layer1.0.", "layer1.")
            state_dict[newkey] = state_dict.pop(key)
        if "encoder.layer2.0." in key:
            newkey = key.replace("encoder.layer2.0.", "layer2.")
            state_dict[newkey] = state_dict.pop(key)
        if "encoder.layer3.0." in key:
            newkey = key.replace("encoder.layer3.0.", "layer3.")
            state_dict[newkey] = state_dict.pop(key)
        if "encoder.layer4.0." in key:
            newkey = key.replace("encoder.layer4.0.", "layer4.")
            state_dict[newkey] = state_dict.pop(key)
        if "classifier" in key:
            state_dict.pop(key)

    msg = model.backbone.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()

def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = [0.0]*config.SAVE_TOP_K_MODEL
    step = 0
    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
    if 'max_accuracy' in checkpoint:
        max_accuracy = checkpoint['max_accuracy']
        logger.info(f"load max_accuracy:{max_accuracy}")
    if 'step' in checkpoint:
        step = checkpoint['step']
        logger.info(f"load step:{step}")

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, step

