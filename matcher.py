# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn



class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self):

        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs.shape[:2] #(bs,num_queries,num_classes)

        # We flatten to compute the cost matrices in a batch
        #softmax变成和为1的概率，softmax适用与多分类问题，sigmoid适用于二分类问题
        out_prob = outputs.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        # Also concat the target labels and boxes
        tgt_ids = targets #[bs,1] -- 图像中真实标注的目标数量
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        #根据真实图片目标数据tgt_ids，从预测的out_prob中选出来
        cost_class = -out_prob[:, tgt_ids] #（bs*queries,bs）

        C =  cost_class #(bs*num_queries,bs)
        # 第一个查询，这6个类的损失。第二个查询，这6个类的损失。。。
        C = C.view(bs, num_queries, -1) #(bs,num_queries,bs).本来是92个obj。只关注tgt给出的那bs个

        sizes = [1 for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        #返回list类型。
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] #返回预测结果


def build_matcher():
    #cost_class=1
    return HungarianMatcher()
