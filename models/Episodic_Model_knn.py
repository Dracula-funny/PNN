
import numpy as np
import pandas as pd
from architectures import get_backbone, get_classifier,get_classifier2
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy
from utils import getprotoconfi,getknnconfi,protoPred,knn_st,statistic,statistic2,addImages,addImages2,class_balance,calculate_accuracy,isThan16,stastic_balance,statistic_balance
import torch
class EpisodicTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = get_backbone(config.MODEL.BACKBONE, *config.MODEL.BACKBONE_HYPERPARAMETERS)
        self.classifier = get_classifier(config.MODEL.CLASSIFIER, *config.MODEL.CLASSIFIER_PARAMETERS)
        self.classifier2 = get_classifier2(config.MODEL.CLASSIFIER, *config.MODEL.CLASSIFIER_PARAMETERS)
        self.train_way = config.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_WAYS
        self.query = config.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_QUERY

        if config.IS_TRAIN == 0:
            self.support = config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_SUPPORT
        else:

            self.support = config.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_SUPPORT
        self.k = config.K
    def forward(self,img_tasks,label_tasks, *args, model, optimizer,step,**kwargs):
        batch_size = len(img_tasks)
        loss = 0.
        loss2 = 0.
        acc = []

        for i, img_task in enumerate(img_tasks):
            support_features = self.backbone(img_task["support"].squeeze_().cuda())
            query_features = self.backbone(img_task["query"].squeeze_().cuda())
            if self.support == 1:
                score2,indices2,knn_distances2,knn_pred2,scores2 = self.classifier(support_features,label_tasks[i]["support"].squeeze_().cuda(),query_features, support_features,
                                        label_tasks[i]["support"].squeeze_().cuda(), self.k-1, **kwargs)
            else:
                score2, indices2, knn_distances2, knn_pred2, scores2 = self.classifier(support_features, label_tasks[i][
                    "support"].squeeze_().cuda(), query_features, support_features,label_tasks[i]["support"].squeeze_().cuda(),self.k, **kwargs)

            proto_pred2 = protoPred(score2, torch.squeeze(label_tasks[i]["support"]))
            knn_trainLabel2 = label_tasks[i]["support"].squeeze_().cuda()
            addImage, addLabel = addImages(proto_pred2,knn_pred2,indices2,knn_trainLabel2)

            labelnum = self.train_way * self.query
            temp = self.support*self.train_way
            addLabel2 = addImages2(support_features,label_tasks[i]["support"].squeeze_().cuda(),query_features,labelnum,temp)

            addLabel2Index = []
            for addLabel2Index1,addLabel2Index2 in enumerate(addLabel2):
                if(addLabel2Index2>-1):
                    addLabel2Index.append(addLabel2Index1)


            add_label = [-1 for _ in range(labelnum)]

            for i5 in range(len(addImage)):
                add_label[addImage[i5]] = addLabel[i5].item()

            add_label3 = add_label.copy()
            for q,w in enumerate(add_label):

                if (w==-1):
                    if (addLabel2[q] <=-1):
                        continue
                    else:
                        add_label3[q] = addLabel2[q]

                else:
                    if(addLabel2[q] <= -1):
                        continue
                    else:
                        if(addLabel2[q] == w):
                            continue
                        else:
                            add_label3[q] = -2

            addImage_pro = []
            addLabel_pro = []

            for x1 in range(len(add_label3)):
                if(add_label3[x1]>-1):
                    addImage_pro.append(x1)
                    addLabel_pro.append(add_label3[x1])


            if len(addImage_pro) != 0:

                proto_support_images = torch.cat((support_features,query_features[addImage_pro]),dim=0)

                addLabel_tensor = torch.tensor(addLabel_pro)
                addLabel_tensor2 = torch.tensor(addLabel_pro)
                proto_support_labels = torch.cat((label_tasks[i]["support"], addLabel_tensor),dim=0).cuda()
            else:
                proto_support_images = support_features
                proto_support_labels = label_tasks[i]["support"].squeeze_().cuda()


            if self.support == 1:
                score3, indices3, knn_distances3, knn_pred3, scores3 = self.classifier(support_features, label_tasks[i][
                    "support"].squeeze_().cuda(), query_features[addImage_pro], support_features, label_tasks[i]["support"].squeeze_().cuda(),self.k-1, **kwargs)
            else:
                score3, indices3, knn_distances3, knn_pred3, scores3 = self.classifier(support_features, label_tasks[i][
                    "support"].squeeze_().cuda(), query_features[addImage_pro], support_features, label_tasks[i][
                                                                                           "support"].squeeze_().cuda(),
                                                                                       self.k, **kwargs)
            knn_trainLabel22 = label_tasks[i]["support"].squeeze_()

            b_tensor = torch.tensor(indices3)
            c_tensor = knn_trainLabel22[b_tensor]
            label_diff2 = torch.abs(c_tensor - addLabel_tensor2.unsqueeze(1).expand(-1, indices3.shape[1]))


            non_zero_count2 = torch.sum(label_diff2 != 0, dim=1)


            knn_loss2 = non_zero_count2.float() / self.k

            knn_loss2 = knn_loss2.mean().item()

            loss2 += (F.cross_entropy(score3, addLabel_tensor2.cuda())) + (knn_loss2)


            score, indices, knn_distances, knn_pred, scores = self.classifier(proto_support_images,proto_support_labels,query_features, support_features,
                                    label_tasks[i]["support"].squeeze_().cuda(), self.k, **kwargs)


            knn_trainLabel = proto_support_labels.cpu()

            label_diff = torch.abs(
                knn_trainLabel[indices] - torch.squeeze(label_tasks[i]["query"]).unsqueeze(1).expand(-1,
                                                                                                     indices.shape[1]))

            non_zero_count = torch.sum(label_diff != 0, dim=1)

            knn_loss = non_zero_count.float() / self.k

            knn_loss = knn_loss.mean().item()


            loss += (F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())) + (knn_loss)


            proto_pred = protoPred(score, torch.squeeze(label_tasks[i]["query"]))

            pre_proto_confi, proto_confi = getprotoconfi(scores, proto_pred)  # 获得原型网络的置信度
            pre_knn_confi, knn_confi = getknnconfi(indices, knn_trainLabel, knn_pred.cpu(), self.k)  # 获得knn的置信度


            knn_st_pred = knn_st(knn_distances, knn_trainLabel, indices, self.train_way * self.query, self.train_way,
                                 self.k)
            new_pred = torch.where(0.7*proto_confi[0] > 0.3*(knn_confi), proto_pred[0], knn_st_pred.to(torch.long))

            new_pred_idx = [i for i in range(75)]

            new_pred3 = new_pred.clone()
            new_pred_idx,new_pred2,updata_idx,proto_confi_stas,knnConfi_stas = class_balance(scores, indices, proto_support_labels, new_pred_idx, new_pred3,self.k)

            new_pred = new_pred2
            acc.append(
                torch.tensor((new_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)


        loss /= batch_size
        model.train()
        loss2 /= batch_size
        loss2.backward()
        optimizer.step()
        model.eval()

        return loss, acc


    def forward2(self,img_tasks,label_tasks, *args, **kwargs):
        batch_size = len(img_tasks)
        loss = 0.
        acc = []
        for i, img_task in enumerate(img_tasks):
            support_features = self.backbone(img_task["support"].squeeze_().cuda())
            query_features = self.backbone(img_task["query"].squeeze_().cuda())
            score2,indices2,knn_distances2,knn_pred2,scores2 = self.classifier2(support_features,label_tasks[i]["support"].squeeze_().cuda(),query_features, support_features,
                                    label_tasks[i]["support"].squeeze_().cuda(), label_tasks[i]["query"].squeeze_().cuda(),self.k, **kwargs)

            proto_pred2 = protoPred(score2, torch.squeeze(label_tasks[i]["support"]))
            knn_trainLabel2 = torch.cat((label_tasks[i]["support"].squeeze_().cuda(), label_tasks[i]["query"].squeeze_().cuda()), dim=0)
            addImage,addLabel = addImages(proto_pred2,knn_pred2,indices2,knn_trainLabel2)


            labelnum = self.train_way * self.query
            temp = self.support * self.train_way
            addLabel2 = addImages2(support_features, label_tasks[i]["support"].squeeze_().cuda(), query_features,
                                   labelnum, temp)

            addLabel2Index = []
            for addLabel2Index1, addLabel2Index2 in enumerate(addLabel2):
                if (addLabel2Index2 > -1):
                    addLabel2Index.append(addLabel2Index1)


            add_label = [-1 for _ in range(labelnum)]

            for i5 in range(len(addImage)):
                add_label[addImage[i5]] = addLabel[i5].item()

            add_label3 = add_label.copy()
            for q, w in enumerate(add_label):

                if (w == -1):
                    if (addLabel2[q] <= -1):
                        continue
                    else:
                        add_label3[q] = addLabel2[q]

                else:
                    if (addLabel2[q] <= -1):
                        continue
                    else:
                        if (addLabel2[q] == w):
                            continue
                        else:
                            add_label3[q] = -2

            addImage_pro = []
            addLabel_pro = []

            for x1 in range(len(add_label3)):
                if (add_label3[x1] > -1):
                    addImage_pro.append(x1)
                    addLabel_pro.append(add_label3[x1])

            if len(addImage_pro) != 0:

                proto_support_images = torch.cat((support_features, query_features[addImage_pro]), dim=0)

                addLabel_tensor = torch.tensor(addLabel_pro)
                proto_support_labels = torch.cat((label_tasks[i]["support"], addLabel_tensor), dim=0).cuda()
            else:
                proto_support_images = support_features
                proto_support_labels = label_tasks[i]["support"].squeeze_().cuda()


            score, indices, knn_distances, knn_pred, scores = self.classifier2(proto_support_images,proto_support_labels,query_features, support_features,
                                                                               label_tasks[i]["support"].squeeze_().cuda(),
                                                                               label_tasks[i]["query"].squeeze_().cuda(),
                                                                               self.k,**kwargs)


            knn_trainLabel = torch.cat((label_tasks[i]["support"], label_tasks[i]["query"]), dim=0)

            label_diff = torch.abs(knn_trainLabel[indices] - torch.squeeze(label_tasks[i]["query"]).unsqueeze(1).expand(-1, indices.shape[1]))

            non_zero_count = torch.sum(label_diff != 0, dim=1)

            knn_loss = non_zero_count.float() / self.k

            knn_loss = knn_loss.mean().item()


            loss += (F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())) + (knn_loss)


            proto_pred = protoPred(score, torch.squeeze(label_tasks[i]["query"]))

            pre_proto_confi,proto_confi = getprotoconfi(scores,proto_pred)
            pre_knn_confi,knn_confi = getknnconfi(indices, knn_trainLabel,knn_pred.cpu(), self.k)



            knn_st_pred = knn_st(knn_distances,knn_trainLabel,indices,self.train_way*self.query,self.train_way,self.k)
            new_pred = torch.where(proto_confi[0] > knn_confi, proto_pred[0], knn_st_pred.to(torch.long))

            acc.append(torch.tensor((new_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item())*100)


        loss /= batch_size
        return loss,acc


    def forward3(self,img_tasks,label_tasks, *args, **kwargs):
        batch_size = len(img_tasks)
        loss = 0.
        acc = []
        for i, img_task in enumerate(img_tasks):
            support_features = self.backbone(img_task["support"].squeeze_().cuda())
            query_features = self.backbone(img_task["query"].squeeze_().cuda())

            if self.support == 1:
                score2,indices2,knn_distances2,knn_pred2,scores2 = self.classifier(support_features,label_tasks[i]["support"].squeeze_().cuda(),query_features, support_features,
                                    label_tasks[i]["support"].squeeze_().cuda(), self.k-1, **kwargs)
            else:
                score2, indices2, knn_distances2, knn_pred2, scores2 = self.classifier(support_features, label_tasks[i][
                    "support"].squeeze_().cuda(), query_features, support_features,label_tasks[i]["support"].squeeze_().cuda(),self.k, **kwargs)
            proto_pred2 = protoPred(score2, torch.squeeze(label_tasks[i]["support"]))
            knn_trainLabel2 = label_tasks[i]["support"].squeeze_().cuda()
            addImage, addLabel = addImages(proto_pred2,knn_pred2,indices2,knn_trainLabel2)

            labelnum = self.train_way * self.query
            temp = self.support * self.train_way
            addLabel2 = addImages2(support_features, label_tasks[i]["support"].squeeze_().cuda(), query_features,
                                   labelnum, temp)

            addLabel2Index = []
            for addLabel2Index1, addLabel2Index2 in enumerate(addLabel2):
                if (addLabel2Index2 > -1):
                    addLabel2Index.append(addLabel2Index1)

            add_label = [-1 for _ in range(labelnum)]

            for i5 in range(len(addImage)):
                add_label[addImage[i5]] = addLabel[i5].item()

            add_label3 = add_label.copy()
            for q, w in enumerate(add_label):

                if (w == -1):
                    if (addLabel2[q] <= -1):
                        continue
                    else:
                        add_label3[q] = addLabel2[q]

                else:
                    if (addLabel2[q] <= -1):
                        continue
                    else:
                        if (addLabel2[q] == w):
                            continue
                        else:
                            add_label3[q] = -2

            addImage_pro = []
            addLabel_pro = []

            for x1 in range(len(add_label3)):
                if (add_label3[x1] > -1):
                    addImage_pro.append(x1)
                    addLabel_pro.append(add_label3[x1])

            if len(addImage_pro) != 0:

                proto_support_images = torch.cat((support_features, query_features[addImage_pro]), dim=0)

                addLabel_tensor = torch.tensor(addLabel_pro)

                proto_support_labels = torch.cat((label_tasks[i]["support"], addLabel_tensor), dim=0).cuda()
            else:
                proto_support_images = support_features
                proto_support_labels = label_tasks[i]["support"].squeeze_().cuda()



            score, indices, knn_distances, knn_pred, scores = self.classifier(proto_support_images,
                                                                              proto_support_labels, query_features,
                                                                              support_features,
                                                                              label_tasks[i][
                                                                                  "support"].squeeze_().cuda(),
                                                                              self.k, **kwargs)

            knn_trainLabel = proto_support_labels.cpu()

            label_diff = torch.abs(
                knn_trainLabel[indices] - torch.squeeze(label_tasks[i]["query"]).unsqueeze(1).expand(-1,
                                                                                                     indices.shape[1]))
            non_zero_count = torch.sum(label_diff != 0, dim=1)
            knn_loss = non_zero_count.float() / self.k
            knn_loss = knn_loss.mean().item()
            loss += (F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())) + (knn_loss)
            proto_pred = protoPred(score, torch.squeeze(label_tasks[i]["support"]))
            pre_proto_confi, proto_confi = getprotoconfi(scores, proto_pred)
            pre_knn_confi, knn_confi = getknnconfi(indices, knn_trainLabel, knn_pred.cpu(), self.k)

            knn_st_pred = knn_st(knn_distances, knn_trainLabel, indices, self.train_way * self.query, self.train_way,
                                 self.k)
            new_pred = torch.where(0.7*proto_confi[0] > (knn_confi)*0.3, proto_pred[0], knn_st_pred.to(torch.long))
            acc.append(
                torch.tensor((new_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)

        loss /= batch_size
        return loss, acc

    def train_forward(self, img_tasks,label_tasks, *args, **kwargs):
        return self.forward2(img_tasks, label_tasks, *args, **kwargs)

    def val_forward(self, img_tasks,label_tasks, *args, **kwargs):

        return self.forward3(img_tasks, label_tasks, *args, **kwargs)

    def test_forward(self, img_tasks,label_tasks, *args, model,optimizer,step,**kwargs):
        return self(img_tasks, label_tasks, *args, model=model,optimizer=optimizer,step = step, **kwargs)

def get_model(config):
    return EpisodicTraining(config)