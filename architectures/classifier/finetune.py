"""
A unified implementation of gradient-based adaptation-time classifiers,
including finetune, URL and cosine classifer.
"""
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import torch.nn as nn
from copy import deepcopy
import math
from .utils import CC_head, prototype_scores
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from .utils import compute_prototypes

class FinetuneModel(torch.nn.Module):
    """
    the overall finetune module that incorporates a backbone and a head.
    """
    def __init__(self, backbone, way, device, use_alpha, use_beta, head):
        super().__init__()
        '''
        backbone: the pre-trained backbone
        way: number of classes
        device: GPU ID
        use_alpha: for TSA only. Whether use adapter to adapt the backbone.
        use_beta: for URL and TSA only. Whether use  pre-classifier transformation.
        head: Use fc head or PN head to finetune.
        '''
        self.backbone = deepcopy(backbone).to(device)
        # if head == "cc":
        #     self.L = CC_head(backbone.outdim, way).to(device)
        # elif head == "fc":
            # self.L = nn.Linear(backbone.outdim, 640).to(device)
            #  self.L = nn.Linear(backbone.outdim, way).to(device)
            # self.L.weight.data.fill_(1)
            # self.L.bias.data.fill_(0)
        self.use_beta = use_beta
        self.head = head

    def forward(self, x, backbone_grad = True):
        # turn backbone_grad off if backbone is not to be finetuned
        if backbone_grad:
            x = self.backbone(x)
        else:
            with torch.no_grad():
                x = self.backbone(x)

        if self.head == "NCC" and self.use_beta:
            x = self.backbone.beta(x)
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, 1).squeeze_(-1).squeeze_(-1)
        x = F.normalize(x, dim=1)
        # if not self.head == "NCC":
        #     x = self.L(x)
        return x

class Finetuner(nn.Module):
    def __init__(self, backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1,ft_lr_2, use_alpha, use_beta, head = "fc"):
        '''
        backbone: the pre-trained backbone
        ft_batchsize: batch size for finetune
        feed_query_batchsize: max number of query images processed once (avoid memory issues)
        ft_epoch: epoch of finetune
        ft_lr_1: backbone learning rate
        ft_lr_2: head learning rate
        use_alpha: for TSA only. Whether use adapter to adapt the backbone.
        use_beta: for URL and TSA only. Whether use  pre-classifier transformation.
        head: the classification head--"fc", "NCC" or "cc"
        '''
        super().__init__()
        self.ft_batchsize = ft_batchsize
        self.feed_query_batchsize = feed_query_batchsize
        self.ft_epoch = ft_epoch
        self.ft_lr_1 = ft_lr_1
        self.ft_lr_2 = ft_lr_2
        self.use_alpha = use_alpha
        self.use_beta = use_beta
        self.head =  head
        self.backbone = backbone

        assert head in ["fc", "NCC", "cc"]

    def forward(self, support_images: Tensor, support_labels,query_images: Tensor) -> Tensor:
        """Take one task of few-shot support examples and query examples as input,
            output the logits of each query examples.

        Args:
            query_images: query examples. size: [num_query, c, h, w]
            support_images: support examples. size: [num_support, c, h, w]
            support_labels: labels of support examples. size: [num_support, way]
        Output:
            classification_scores: The calculated logits of query examples.
                                   size: [num_query, way]
        """
        support_size = support_images.size(0)

        device = support_images.device

        way = torch.max(support_labels).item()+1
        model = FinetuneModel(self.backbone, way, device, self.use_alpha, self.use_beta, self.head)

        # By default, SGD is adopted as the optimizer. Other optimizers like Adam can be used as well.

        if self.ft_lr_1==0.:
            # URL or Linear classifier or cosine classifier. No backbone tuning needed.
            set_optimizer_1 = None
        elif self.head == "NCC" and self.use_alpha:
            # if using adapter, then the adapters are tuned
            alpha_params = [v for k, v in model.backbone.named_parameters() if 'alpha' in k]
            set_optimizer_1 = torch.optim.SGD(alpha_params, lr = self.ft_lr_1, momentum=0.9)
        else:
            set_optimizer_1 = torch.optim.SGD(model.backbone.parameters(), lr = self.ft_lr_1, momentum=0.9)

        # if self.head == "NCC" and self.use_beta:
        #     beta_params = [v for k, v in model.backbone.named_parameters() if 'beta' in k]
        #     set_optimizer_2 = torch.optim.SGD(beta_params, lr = self.ft_lr_2, momentum=0.9)
        # elif not self.head == "NCC":
        #     set_optimizer_2 = torch.optim.SGD(model.L.parameters(), lr = self.ft_lr_2, momentum=0.9)
        # else:
        #     # If using NCC as the tuning loss and no pre-classifier transformation is used,
        #     # then there is no head to be learned.
        #     set_optimizer_2 = None

        model.eval()

        # total finetuning steps
        global_steps = self.ft_epoch*((support_size+self.ft_batchsize-1)//self.ft_batchsize)

        step = 0

        with torch.enable_grad():
            for epoch in range(self.ft_epoch):
                # randomly suffule support set
                rand_id = np.random.permutation(support_size)

                for i in range(0, support_size , self.ft_batchsize):
                    # by default, cosine LR shedule is used.
                    lr_1 = 0.5 * self.ft_lr_1* (1. + math.cos(math.pi * step / global_steps))
                    lr_2 = 0.5 * self.ft_lr_2* (1. + math.cos(math.pi * step / global_steps))
                    if set_optimizer_1 is not None:
                        for param_group in set_optimizer_1.param_groups:
                            param_group["lr"] = lr_1
                        set_optimizer_1.zero_grad()
                    # if set_optimizer_2 is not None:
                    #     for param_group in set_optimizer_2.param_groups:
                    #         param_group["lr"] = lr_2
                    #     set_optimizer_2.zero_grad()


                    selected_id = torch.from_numpy(rand_id[i: min(i+self.ft_batchsize, support_size)])
                    train_batch = support_images[selected_id]
                    label_batch = support_labels[selected_id]

                    if set_optimizer_1 is not None:
                        train_batch = model(train_batch)
                    else:
                        train_batch = model(train_batch, backbone_grad = False)

                    if not self.head == "NCC":
                        loss = F.cross_entropy(train_batch, label_batch)
                    else:

                        nearest_neighbors = NearestNeighbors(n_neighbors=6, metric='cosine').fit(
                            train_batch.cpu().detach().numpy())
                        distances, indices = nearest_neighbors.kneighbors(train_batch.cpu().detach().numpy())
                        rows, cols = distances.shape
                        distances2 = np.zeros((rows, cols - 1))
                        indices2 = np.zeros((rows, cols - 1))
                        # 遍历每一行，按照条件裁剪
                        for i in range(distances.shape[0]):
                            if round(distances[i, 0], 6) == 0:
                                distances2[i, :] = distances[i, 1:6]
                                indices2[i, :] = indices[i, 1:6]
                            else:
                                distances2[i, :] = distances[i, 0:5]
                                indices2[i, :] = indices[i, 0:5]

                        res = torch.empty(0, dtype=torch.int, device='cuda')
                        for idxs in indices2:
                            # voting
                            # 代码下方会附上解释np.bincount()函数的博客
                            result = torch.bincount(
                                torch.tensor([label_batch[idx] for idx in idxs[:5]], dtype=torch.int,
                                             device='cuda')).argmax()

                            res = torch.cat((res, result.unsqueeze(0).to(torch.int)))
                        assert len(res) == len(query_images)
                        test_predictions = res



                        knn_trainLabel = label_batch.cpu()
                        # knn_trainLabel = torch.cat((label_tasks[i]["support"], label_tasks[i]["query"].squeeze_()), dim=0)
                        # 计算标签差异值并求绝对值---------------这行有问题，label_tasks[i]["query"]和原来代码里的不一样
                        label_diff = torch.abs(
                            knn_trainLabel[indices2] - torch.squeeze(knn_trainLabel).unsqueeze(1).expand(-1,indices2.shape[1]))
                        # 计算非零元素的数量
                        non_zero_count = torch.sum(label_diff != 0, dim=1)
                        # 计算损失并进行指数运算
                        # knn_loss = torch.exp(non_zero_count.float() / self.k)
                        knn_loss = non_zero_count.float() / 5
                        # 计算平均损失
                        knn_loss = knn_loss.mean().item()



                        score = prototype_scores(train_batch, label_batch,
                                       train_batch)
                        loss = F.cross_entropy(score, label_batch)+knn_loss

                    loss.backward()
                    if set_optimizer_1 is not None:
                        set_optimizer_1.step()
                    # if set_optimizer_2 is not None:
                    #     set_optimizer_2.step()
                    # step += 1

        model.eval()

        # number of feed-forward calculations to calculate all query embeddings
        query_runs = (query_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize

        # if not self.head == "NCC":
        if  self.head == "NCC":
            scores = []
            for run in range(query_runs):
                # for non-NCC head, the model directly ouputs score
                scores.append(model(query_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,query_images.size(0))]))
            classification_scores = torch.cat(scores, dim=0)
        else:
            support_features = []
            query_features = []
            # number of feed-forward calculations to calculate all support embeddings
            support_runs = (support_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize
            for run in range(support_runs):
                support_features.append(model(support_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,support_images.size(0))]))
            for run in range(query_runs):
                query_features.append(model(query_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,query_images.size(0))]))
            support_features = torch.cat(support_features, dim=0)
            query_features = torch.cat(query_features, dim=0)
            classification_scores = prototype_scores(support_features, support_labels,
                            query_features)

            nearest_neighbors = NearestNeighbors(n_neighbors=6, metric='cosine').fit(
                support_features.cpu().detach().numpy())
            distances, indices = nearest_neighbors.kneighbors(query_features.cpu().detach().numpy())
            rows, cols = distances.shape
            distances2 = np.zeros((rows, cols - 1))
            indices2 = np.zeros((rows, cols - 1))
            # 遍历每一行，按照条件裁剪
            for i in range(distances.shape[0]):
                if round(distances[i, 0], 6) == 0:
                    distances2[i, :] = distances[i, 1:6]
                    indices2[i, :] = indices[i, 1:6]
                else:
                    distances2[i, :] = distances[i, 0:5]
                    indices2[i, :] = indices[i, 0:5]

            res = torch.empty(0, dtype=torch.int, device='cuda')
            indices2 = indices2.astype(int)
            for idxs in indices2:
                # voting
                # 代码下方会附上解释np.bincount()函数的博客
                result = torch.bincount(
                    torch.tensor([support_labels[idx] for idx in idxs[:5]], dtype=torch.int,
                                 device='cuda')).argmax()

                res = torch.cat((res, result.unsqueeze(0).to(torch.int)))
            assert len(res) == len(query_images)
            test_predictions = res




        return classification_scores, indices2,distances2, test_predictions,classification_scores

def create_model(backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1,ft_lr_2, use_alpha, use_beta, head = 'fc'):
    return Finetuner(backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1,ft_lr_2, use_alpha, use_beta, head)


class PN_head(nn.Module):
    def __init__(self,
                 scale_cls: int =10.0,
                 learn_scale: bool = False,
                 knn_confi_weight = 1.0,
                 proto_confi_weight = 1.0) -> None:
        super().__init__()
        if learn_scale:
            self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)
            # self.knn_confi_weight = nn.Parameter(torch.FloatTensor(10).fill_(knn_confi_weight), requires_grad=True)
            # self.proto_confi_weight = nn.Parameter(torch.FloatTensor(10).fill_(proto_confi_weight), requires_grad=True)
        else:
            self.scale_cls = scale_cls
            # self.knn_confi_weight = knn_confi_weight
            # self.proto_confi_weight = proto_confi_weight

    def forward(self, proto_support_images: Tensor,proto_support_labels,query_images: Tensor) -> Tensor:
        """Take one task of few-shot support examples and query examples as input,
            output the logits of each query examples.

        Args:
            query_images: query examples. size: [num_query, c, h, w]
            support_images: support examples. size: [num_support, c, h, w]
            support_labels: labels of support examples. size: [num_support, way]
        Output:
            classification_scores: The calculated logits of query examples.
                                   size: [num_query, way]
        """
        if query_images.dim() == 4:

            proto_support_images = F.adaptive_avg_pool2d(proto_support_images, 1).squeeze_(-1).squeeze_(-1)
            query_images = F.adaptive_avg_pool2d(query_images, 1).squeeze_(-1).squeeze_(-1)

        assert query_images.dim() == proto_support_images.dim() == 2



        query_images = F.normalize(query_images, p=2, dim=1, eps=1e-12)
        proto_support_images = F.normalize(proto_support_images, p=2, dim=1, eps=1e-12)

        one_hot_label = F.one_hot(proto_support_labels,num_classes = torch.max(proto_support_labels).item()+1).float()


        nearest_neighbors = NearestNeighbors(n_neighbors=5,metric='cosine').fit(proto_support_images.cpu().detach().numpy())
        distances, indices = nearest_neighbors.kneighbors(query_images.cpu().detach().numpy())


        res = torch.empty(0, dtype=torch.int, device='cuda')
        for idxs in indices:
            # voting
            # 代码下方会附上解释np.bincount()函数的博客
            result = torch.bincount(
                torch.tensor([proto_support_labels[idx] for idx in idxs[:5]], dtype=torch.int, device='cuda')).argmax()

            res = torch.cat((res, result.unsqueeze(0).to(torch.int)))
        assert len(res) == len(query_images)
        test_predictions = res

        #prototypes: [way, c]
        prototypes = compute_prototypes(proto_support_images, one_hot_label)

        prototypes = F.normalize(prototypes, p=2, dim=1, eps=1e-12)

        classification_scores = 10*torch.mm(query_images, prototypes.transpose(0, 1))
        scores = torch.mm(query_images, prototypes.transpose(0, 1))
        #classification_scores = euclidean_metric(query_images, prototypes)
        # return classification_scores,indices,test_predictions,scores,self.knn_confi_weight,self.proto_confi_weight

        return classification_scores, indices,distances, test_predictions, scores


def create_model2():
    return PN_head()