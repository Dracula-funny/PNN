"""
module for gradient-based test-time methods, e.g., finetune, eTT, TSA, URL, Cosine Classifier
"""
from architectures import get_backbone, get_classifier,get_classifier2
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy
from utils import getprotoconfi,getknnconfi,protoPred,knn_st,statistic,statistic2,addImages
import torch
class FinetuneModule(nn.Module):
    def __init__(self, config):
        super().__init__()


        self.classifier = get_classifier(config.MODEL.CLASSIFIER, *config.MODEL.CLASSIFIER_PARAMETERS)
        self.classifier2 = get_classifier2(config.MODEL.CLASSIFIER)
        self.train_way = config.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_WAYS
        self.query = config.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_QUERY
        self.k = config.K


        self.config = config
        self.backbone = get_backbone(config.MODEL.BACKBONE, *config.MODEL.BACKBONE_HYPERPARAMETERS)

        # The last hyperparameter is the head mode
        self.mode = config.MODEL.CLASSIFIER_PARAMETERS[-1]

        
        if not self.mode == "NCC":
            classifier_hyperparameters = [self.backbone]+config.MODEL.CLASSIFIER_PARAMETERS
            self.classifier = get_classifier(config.MODEL.CLASSIFIER, *classifier_hyperparameters)
            self.classifier2 = get_classifier2(self.config.MODEL.CLASSIFIER)
    def append_adapter(self):
        # append adapter to the backbone
        self.backbone = get_backbone("resnet_tsa",backbone=self.backbone)
        classifier_hyperparameters = [self.backbone]+self.config.MODEL.CLASSIFIER_PARAMETERS
        self.classifier = get_classifier(self.config.MODEL.CLASSIFIER, *classifier_hyperparameters)

        self.classifier2 = get_classifier2(self.config.MODEL.CLASSIFIER)
    def test_forward(self, img_tasks,label_tasks, *args, **kwargs):
        batch_size = len(img_tasks)
        loss = 0.

        knn_acc = []
        proto_acc = []
        acc = []
        for i, img_task in enumerate(img_tasks):

            support_features = self.backbone(img_task["support"].squeeze_().cuda())
            query_features = self.backbone(img_task["query"].squeeze_().cuda())
            score2,indices2,knn_distances2,knn_pred2,scores2 = self.classifier2(support_features,label_tasks[i]["support"].squeeze_().cuda(),query_features)

            proto_pred2 = protoPred(score2, torch.squeeze(label_tasks[i]["query"]))
            knn_trainLabel2 = label_tasks[i]["support"].squeeze_().cuda()
            addImage, addLabel = addImages(proto_pred2, knn_pred2, indices2, knn_trainLabel2)

            if len(addImage) != 0:

                proto_support_images = torch.cat((img_task["support"].squeeze_().cuda(), (img_task["query"].squeeze_().cuda())[addImage]), dim=0)

                addLabel_tensor = torch.tensor(addLabel)
                proto_support_labels = torch.cat((label_tasks[i]["support"], addLabel_tensor), dim=0).cuda()
            else:
                proto_support_images = img_task["support"].squeeze_().cuda()
                proto_support_labels = label_tasks[i]["support"].squeeze_().cuda()


            score, indices, knn_distances, knn_pred, scores = self.classifier(proto_support_images,
                                    proto_support_labels,img_task["query"].squeeze_().cuda())
            # score = self.classifier(img_task["query"].squeeze_().cuda(), img_task["support"].squeeze_().cuda(),
            #                         label_tasks[i]["support"].squeeze_().cuda())

            knn_trainLabel = proto_support_labels.cpu()
            # knn_trainLabel = torch.cat((label_tasks[i]["support"], label_tasks[i]["query"].squeeze_()), dim=0)
            # 计算标签差异值并求绝对值---------------这行有问题，label_tasks[i]["query"]和原来代码里的不一样
            label_diff = torch.abs(
                knn_trainLabel[indices] - torch.squeeze(label_tasks[i]["query"]).unsqueeze(1).expand(-1,
                                                                                                     indices.shape[1]))
            # 计算非零元素的数量
            non_zero_count = torch.sum(label_diff != 0, dim=1)
            # 计算损失并进行指数运算
            # knn_loss = torch.exp(non_zero_count.float() / self.k)
            knn_loss = non_zero_count.float() / 5
            # 计算平均损失
            knn_loss = knn_loss.mean().item()



            proto_pred = protoPred(score, torch.squeeze(label_tasks[i]["query"]))

            pre_proto_confi, proto_confi = getprotoconfi(scores, proto_pred)  # 获得原型网络的置信度
            pre_knn_confi, knn_confi = getknnconfi(indices, knn_trainLabel, knn_pred.cpu(), 5)  # 获得knn的置信度

            knn_st_pred = knn_st(knn_distances, knn_trainLabel, indices, self.train_way * self.query, self.train_way,
                                 5)
            new_pred = torch.where(0.63 * proto_confi[0] > (knn_confi) * 0.37, proto_pred[0],
                                   knn_st_pred.to(torch.long))

            loss += F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda()) + knn_loss

            acc.append(
                torch.tensor((new_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            knn_acc.append(
                torch.tensor((knn_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            proto_acc.append(torch.tensor(
                (proto_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)


            # acc.append(accuracy(score, label_tasks[i]["query"].cuda())[0])
        loss /= batch_size
        return loss, knn_acc, proto_acc, acc

def get_model(config):
    return FinetuneModule(config)