import mindspore.ops as ops
import mindspore as ms
import mindspore.nn as nn
import mindspore.common.dtype as mstype

import os
from .utils import concat_box_prediction_layers
from .model_utils import smooth_l1_loss
from .model_utils.sigmoid_focal_loss import SigmoidFocalClassificationLoss
from .model_utils.matcher import Matcher
# from paa_core.structures.boxlist_ops import boxlist_iou
from .model_utils.boxlist_ops import cat_boxlist
from .model_utils.bounding_box import BoxList
from .sklearn.mixture._gaussian_mixture import GaussianMixture



def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    op=ops.Concat(dim)
    return op(tensors)

def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes

def boxlist_iou(boxlist1, boxlist2):
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def min(a,b):
    if a<b:
        return a
    return b

def max(a,b):
    if a<b:
        return a
    return b
INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


class PAALossComputation(object):

    def __init__(self, cfg, box_coder):
        self.cfg = cfg
        self.cls_loss_func = SigmoidFocalClassificationLoss(cfg.MODEL.PAA.LOSS_GAMMA,
                                              cfg.MODEL.PAA.LOSS_ALPHA)
        self.iou_pred_loss_func =nn.BCEWithLogitsLoss(reduction="sum")
        self.matcher = Matcher(cfg.MODEL.PAA.IOU_THRESHOLD,
                               cfg.MODEL.PAA.IOU_THRESHOLD,
                               True)
        self.box_coder = box_coder
        self.fpn_strides=[8, 16, 32, 64, 128]
        self.reg_loss_type = cfg.MODEL.PAA.REG_LOSS_TYPE
        self.iou_loss_weight = cfg.MODEL.PAA.IOU_LOSS_WEIGHT

    def GIoULoss(self, pred, target, anchor, weight=None):
        pred_boxes = self.box_coder.decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        if pred_area<0:
            pred_area=-pred_area

        gt_boxes = self.box_coder.decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = max(pred_x1, target_x1)
        y1_intersect = max(pred_y1, target_y1)
        x2_intersect = min(pred_x2, target_x2)
        y2_intersect = min(pred_y2, target_y2)
        area_intersect = ops.Zeros(pred_x1.size(), mstype.float32).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = min(pred_x1, target_x1)
        y1_enclosing = min(pred_y1, target_y1)
        x2_enclosing = max(pred_x2, target_x2)
        y2_enclosing = max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight)
        else:
            assert losses.numel() != 0
            return losses

    def prepare_iou_based_targets(self, targets, anchors):
        """Compute IoU-based targets"""

        cls_labels = []
        reg_targets = []
        matched_idx_all = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes_per_im = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            anchors_per_im = cat_boxlist(anchors[im_i])
            num_gt = bboxes_per_im.shape[0]

            match_quality_matrix = boxlist_iou(targets_per_im, anchors_per_im)
            matched_idxs = self.matcher(match_quality_matrix)
            targets_per_im = targets_per_im.copy_with_fields(['labels'])
            matched_targets = targets_per_im[matched_idxs.clamp(min=0)]

            cls_labels_per_im = matched_targets.get_field("labels")
            cls_labels_per_im = cls_labels_per_im.to(dtype=mstype.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            cls_labels_per_im[bg_indices] = 0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            cls_labels_per_im[inds_to_discard] = -1

            matched_gts = matched_targets.bbox
            matched_idx_all.append(matched_idxs.view(1, -1))

            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return cls_labels, reg_targets, matched_idx_all

    def compute_paa(self, targets, anchors, labels_all, loss_all, matched_idx_all):
        """
        Args:
            targets (batch_size): list of BoxLists for GT bboxes
            anchors (batch_size, feature_lvls): anchor boxes per feature level
            labels_all (batch_size x num_anchors): assigned labels
            loss_all (batch_size x num_anchors): calculated loss
            matched_idx_all (batch_size x num_anchors): best-matched GG bbox indexes
        """
        device = loss_all.device
        cls_labels = []
        reg_targets = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes_per_im = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            anchors_per_im = cat_boxlist(anchors[im_i])
            labels_all_per_im = labels_all[im_i]
            loss_all_per_im = loss_all[im_i]
            matched_idx_all_per_im = matched_idx_all[im_i]
            assert labels_all_per_im.shape == matched_idx_all_per_im.shape

            num_anchors_per_level = [len(anchors_per_level.bbox)
                for anchors_per_level in anchors[im_i]]

            # select candidates based on IoUs between anchors and GTs
            candidate_idxs = []
            num_gt = bboxes_per_im.shape[0]
            for gt in range(num_gt):
                candidate_idxs_per_gt = []
                star_idx = 0
                for level, anchors_per_level in enumerate(anchors[im_i]):
                    end_idx = star_idx + num_anchors_per_level[level]
                    loss_per_level = loss_all_per_im[star_idx:end_idx]
                    labels_per_level = labels_all_per_im[star_idx:end_idx]
                    matched_idx_per_level = matched_idx_all_per_im[star_idx:end_idx]
                    match_idx = ops.nonzero(
                        (matched_idx_per_level == gt) & (labels_per_level > 0),
                        as_tuple=False
                    )[:, 0]
                    if match_idx.numel() > 0:
                        _, topk_idxs = loss_per_level[match_idx].topk(
                            min(match_idx.numel(), self.cfg.MODEL.PAA.TOPK), largest=False)
                        topk_idxs_per_level_per_gt = match_idx[topk_idxs]
                        candidate_idxs_per_gt.append(topk_idxs_per_level_per_gt + star_idx)
                    star_idx = end_idx
                if candidate_idxs_per_gt:
                    op=ops.Concat()
                    candidate_idxs.append(op(candidate_idxs_per_gt))
                else:
                    candidate_idxs.append(None)

            # fit 2-mode GMM per GT box
            n_labels = anchors_per_im.bbox.shape[0]
            cls_labels_per_im = ops.Zeros(n_labels, dtype=mstype.float64).to(device)
            matched_gts = ops.ZerosLike(anchors_per_im.bbox)
            fg_inds = matched_idx_all_per_im >= 0
            matched_gts[fg_inds] = bboxes_per_im[matched_idx_all_per_im[fg_inds]]
            is_grey = None
            for gt in range(num_gt):
                if candidate_idxs[gt] is not None:
                    if candidate_idxs[gt].numel() > 1:
                        candidate_loss = loss_all_per_im[candidate_idxs[gt]]
                        candidate_loss, inds = candidate_loss.sort()
                        candidate_loss = candidate_loss.view(-1, 1).cpu().numpy()
                        min_loss, max_loss = candidate_loss.min(), candidate_loss.max()
                        means_init=[[min_loss], [max_loss]]
                        weights_init = [0.5, 0.5]
                        precisions_init=[[[1.0]], [[1.0]]]
                        gmm = GaussianMixture(2,weights_init=weights_init,means_init=means_init,
                                                  precisions_init=precisions_init)
                        gmm.fit(candidate_loss)
                        components = gmm.predict(candidate_loss)
                        scores = gmm.score_samples(candidate_loss)
                        components = ms.tensor.from_numpy(components).to(device)
                        scores = ms.tensor.from_numpy(scores).to(device)
                        fgs = components == 0
                        bgs = components == 1
                        if ops.nonzero(fgs, as_tuple=False).numel() > 0:
                            # Fig 3. (c)
                            fg_max_score = scores[fgs].max().item()
                            fg_max_idx = ops.nonzero(fgs & (scores == fg_max_score), as_tuple=False).min()
                            is_neg = inds[fgs | bgs]
                            is_pos = inds[:fg_max_idx+1]
                        else:
                            # just treat all samples as positive for high recall.
                            is_pos = inds
                            is_neg = is_grey = None
                    else:
                        is_pos = 0
                        is_neg = None
                        is_grey = None
                    if is_grey is not None:
                        grey_idx = candidate_idxs[gt][is_grey]
                        cls_labels_per_im[grey_idx] = -1
                    if is_neg is not None:
                        neg_idx = candidate_idxs[gt][is_neg]
                        cls_labels_per_im[neg_idx] = 0
                    pos_idx = candidate_idxs[gt][is_pos]
                    cls_labels_per_im[pos_idx] = labels_per_im[gt].view(-1, 1)
                    matched_gts[pos_idx] = bboxes_per_im[gt].view(-1, 4)

            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return cls_labels, reg_targets

    def compute_reg_loss(
        self, regression_targets, box_regression, all_anchors, labels, weights
    ):
        if 'iou' in self.reg_loss_type:
            reg_loss = self.GIoULoss(box_regression,
                                     regression_targets,
                                     all_anchors,
                                     weight=weights)
        elif self.reg_loss_type == 'smoothl1':
            reg_loss = smooth_l1_loss(box_regression,
                                      regression_targets,
                                      beta=self.bbox_reg_beta,
                                      size_average=False,
                                      sum=False)
            if weights is not None:
                reg_loss *= weights
        else:
            raise NotImplementedError
        return reg_loss[labels > 0].view(-1)

    def compute_ious(self, boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
        area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
        lt = max(boxes1[:, :2], boxes2[:, :2])
        rb = min(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        return inter / (area1 + area2 - inter)

    def __call__(self, box_cls, box_regression, iou_pred, targets, anchors, locations):

        # get IoU-based anchor assignment first to compute anchor scores
        (iou_based_labels,
         iou_based_reg_targets,
         matched_idx_all) = self.prepare_iou_based_targets(targets, anchors)
        op=ops.Concat(0)
        matched_idx_all = op(matched_idx_all)

        N = len(iou_based_labels)
        iou_based_labels_flatten = op(iou_based_labels).int()
        iou_based_reg_targets_flatten = op(iou_based_reg_targets)
        box_cls_flatten, box_regression_flatten = concat_box_prediction_layers(
            box_cls, box_regression)
        anchors_flatten = op([cat_boxlist(anchors_per_image).bbox
            for anchors_per_image in anchors])
        if iou_pred is not None:
            iou_pred_flatten = [ip.permute(0, 2, 3, 1).reshape(N, -1, 1) for ip in iou_pred]
            op=ops.Concat(1)
            iou_pred_flatten = op(iou_pred_flatten).reshape(-1)

        pos_inds = ops.nonzero(iou_based_labels_flatten > 0, as_tuple=False).squeeze(1)

        if pos_inds.numel() > 0:
            n_anchors = box_regression[0].shape[1] // 4
            n_loss_per_box = 1 if 'iou' in self.reg_loss_type else 4

            # compute anchor scores (losses) for all anchors
            iou_based_cls_loss = self.cls_loss_func(box_cls_flatten.detach(),
                                                    iou_based_labels_flatten,
                                                    sum=False)
            iou_based_reg_loss = self.compute_reg_loss(iou_based_reg_targets_flatten,
                                                       box_regression_flatten.detach(),
                                                       anchors_flatten,
                                                       iou_based_labels_flatten,
                                                       weights=None)
            iou_based_reg_loss_full = ms.numpy.full((iou_based_cls_loss.shape[0],),
                                                  fill_value=INF,
                                                  device=iou_based_cls_loss.device,
                                                  dtype=iou_based_cls_loss.dtype)
            iou_based_reg_loss_full[pos_inds] = iou_based_reg_loss.view(-1, n_loss_per_box).mean(1)
            combined_loss = iou_based_cls_loss.sum(dim=1) + iou_based_reg_loss_full

            # compute labels and targets using PAA
            labels, reg_targets = self.compute_paa(
                targets,
                anchors,
                iou_based_labels_flatten.view(N, -1),
                combined_loss.view(N, -1),
                matched_idx_all)
            op=ops.Concat()
            num_gpus = get_num_gpus()
            labels_flatten = op(labels).int()
            reg_targets_flatten = op(reg_targets)
            pos_inds = ops.nonzero(labels_flatten > 0, as_tuple=False).squeeze(1)
            total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
            num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

            box_regression_flatten = box_regression_flatten[pos_inds]
            reg_targets_flatten = reg_targets_flatten[pos_inds]
            anchors_flatten = anchors_flatten[pos_inds]

            if iou_pred is not None:
                # compute iou prediction targets
                iou_pred_flatten = iou_pred_flatten[pos_inds]
                gt_boxes = self.box_coder.decode(reg_targets_flatten, anchors_flatten)
                boxes = self.box_coder.decode(box_regression_flatten, anchors_flatten).detach()
                ious = self.compute_ious(gt_boxes, boxes)

                # compute iou losses
                iou_pred_loss = self.iou_pred_loss_func(
                    iou_pred_flatten, ious) / num_pos_avg_per_gpu * self.iou_loss_weight
                sum_ious_targets_avg_per_gpu = reduce_sum(ious.sum()).item() / float(num_gpus)

                # set regression loss weights to ious between predicted boxes and GTs
                reg_loss_weight = ious
            else:
                reg_loss_weight = None

            reg_loss = self.compute_reg_loss(reg_targets_flatten,
                                             box_regression_flatten,
                                             anchors_flatten,
                                             labels_flatten[pos_inds],
                                             weights=reg_loss_weight)
            cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten.int(), sum=False)
        else:
            reg_loss = box_regression_flatten.sum()

        reg_norm = sum_ious_targets_avg_per_gpu if iou_pred is not None else num_pos_avg_per_gpu
        res = [cls_loss.sum() / num_pos_avg_per_gpu,
               reg_loss.sum() / reg_norm * self.cfg.MODEL.PAA.REG_LOSS_WEIGHT]
        if iou_pred is not None:
            res.append(iou_pred_loss)
        return res


def make_paa_loss_evaluator(cfg, box_coder):
    loss_evaluator = PAALossComputation(cfg, box_coder)
    return loss_evaluator
