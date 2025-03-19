"""
The metric-based protypical classifier (Nearest-Centroid Classifier) from ``Prototypical Networks for Few-shot Learning''.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import compute_prototypes,euclidean_metric
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

class PN_head(nn.Module):
    def __init__(self,
                 scale_cls: int =10.0, 
                 learn_scale: bool = True) -> None:
        super().__init__()
        if learn_scale:
            self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)

        else:
            self.scale_cls = scale_cls


    def forward(self, proto_support_images: Tensor,proto_support_labels,query_images: Tensor, support_images: Tensor, support_labels,k) -> Tensor:
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
            support_images = F.adaptive_avg_pool2d(support_images, 1).squeeze_(-1).squeeze_(-1)
            proto_support_images = F.adaptive_avg_pool2d(proto_support_images, 1).squeeze_(-1).squeeze_(-1)
            query_images = F.adaptive_avg_pool2d(query_images, 1).squeeze_(-1).squeeze_(-1)

        assert support_images.dim() == query_images.dim() == proto_support_images.dim() == 2

        query_images = F.normalize(query_images, p=2, dim=1, eps=1e-12)
        proto_support_images = F.normalize(proto_support_images, p=2, dim=1, eps=1e-12)

        one_hot_label = F.one_hot(proto_support_labels,num_classes = torch.max(proto_support_labels).item()+1).float()
        nearest_neighbors = NearestNeighbors(n_neighbors=k+1,metric='cosine').fit(proto_support_images.cpu().detach().numpy())
        distances, indices = nearest_neighbors.kneighbors(query_images.cpu().detach().numpy())
        rows, cols = distances.shape
        distances2 = np.zeros((rows, cols-1))
        indices2 = np.zeros((rows, cols-1))
        if (k == 5):
            for i in range(distances.shape[0]):
                if round(distances[i, 0], 6) == 0:
                    distances2[i, :] = distances[i, 1:6]
                    indices2[i, :] = indices[i, 1:6]
                else:
                    distances2[i, :] = distances[i, 0:5]
                    indices2[i, :] = indices[i, 0:5]
        else:
            distances2 = distances
            indices2 = indices
        res = torch.empty(0, dtype=torch.int, device='cuda')
        indices2 = indices2.astype(int)
        for idxs in indices2:
            result = torch.bincount(
                torch.tensor([proto_support_labels[idx] for idx in idxs[:k]], dtype=torch.int, device='cuda')).argmax()

            res = torch.cat((res, result.unsqueeze(0).to(torch.int)))
        assert len(res) == len(query_images)
        test_predictions = res
        prototypes = compute_prototypes(proto_support_images, one_hot_label)

        prototypes = F.normalize(prototypes, p=2, dim=1, eps=1e-12)

        classification_scores = self.scale_cls*torch.mm(query_images, prototypes.transpose(0, 1))
        scores = torch.mm(query_images, prototypes.transpose(0, 1))

        return classification_scores, indices2,distances2, test_predictions, scores


def create_model():
    return PN_head()

class PN_head2(nn.Module):
    def __init__(self,
                 scale_cls: int =10.0,
                 learn_scale: bool = True) -> None:
        super().__init__()
        if learn_scale:
            self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)

        else:
            self.scale_cls = scale_cls

    def forward(self, proto_support_images: Tensor,proto_support_labels,query_images: Tensor, support_images: Tensor, support_labels,query_labels,k) -> Tensor:

        if query_images.dim() == 4:
            support_images = F.adaptive_avg_pool2d(support_images, 1).squeeze_(-1).squeeze_(-1)
            proto_support_images = F.adaptive_avg_pool2d(proto_support_images, 1).squeeze_(-1).squeeze_(-1)
            query_images = F.adaptive_avg_pool2d(query_images, 1).squeeze_(-1).squeeze_(-1)

        assert support_images.dim() == query_images.dim() == proto_support_images.dim() == 2

        support_images = F.normalize(support_images, p=2, dim=1, eps=1e-12)
        query_images = F.normalize(query_images, p=2, dim=1, eps=1e-12)

        one_hot_label = F.one_hot(proto_support_labels,num_classes = torch.max(proto_support_labels).item()+1).float()

        knn_trainImage = torch.cat((support_images, query_images), dim=0)
        knn_trainLabel = torch.cat((support_labels, query_labels), dim=0)
        nearest_neighbors = NearestNeighbors(n_neighbors=k+1,metric='cosine').fit(knn_trainImage.cpu().detach().numpy())
        distances, indices = nearest_neighbors.kneighbors(query_images.cpu().detach().numpy())
        indices =indices[:,1:]
        distances = distances[:,1:]


        res = torch.empty(0, dtype=torch.int, device='cuda')
        for idxs in indices:

            result = torch.bincount(
                torch.tensor([knn_trainLabel[idx] for idx in idxs[:k]], dtype=torch.int, device='cuda')).argmax()

            res = torch.cat((res, result.unsqueeze(0).to(torch.int)))
        assert len(res) == len(query_images)
        test_predictions = res


        prototypes = compute_prototypes(proto_support_images, one_hot_label)

        prototypes = F.normalize(prototypes, p=2, dim=1, eps=1e-12)

        classification_scores = self.scale_cls*torch.mm(query_images, prototypes.transpose(0, 1))
        scores = torch.mm(query_images, prototypes.transpose(0, 1))

        return classification_scores, indices,distances, test_predictions, scores


def create_model2():
    return PN_head2()