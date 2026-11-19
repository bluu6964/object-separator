"""Class-agnostic box + mask mAP@0.5 evaluation and NMS. Only torch/numpy.

Matching rule (single "object" class): detections sorted by score; a
detection is a TP if its IoU (box IoU for box AP, mask IoU for mask AP)
with an unmatched GT instance is >= 0.5. AP is all-point interpolated.
"""
import numpy as np
import torch

from model import IMG_SIZE, PROTO_SIZE, decode


def nms(boxes, scores, iou_thres=0.5):
    """Greedy NMS. boxes [N,4] xyxy, scores [N]. Returns kept indices (list)."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thres]
    return keep


def box_iou_one_to_many(box, boxes):
    """IoU of one box [4] against many [N,4] (numpy)."""
    if len(boxes) == 0:
        return np.zeros(0)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    a1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    a2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a1 + a2 - inter, 1e-9)


def mask_iou_one_to_many(mask, masks):
    """IoU of one bool mask [P,P] against many [N,P,P] (numpy bool)."""
    if len(masks) == 0:
        return np.zeros(0)
    m = np.asarray(mask, dtype=bool)
    ms = np.asarray(masks, dtype=bool)
    inter = (ms & m).sum(axis=(1, 2)).astype(np.float64)
    union = (ms | m).sum(axis=(1, 2)).astype(np.float64)
    return inter / np.maximum(union, 1e-9)


def _ap_from_matches(dets, n_gt, iou_fn):
    """dets: list of (score, image_idx, payload) sorted internally.
    iou_fn(payload, gt_payloads_for_image) -> IoU array."""
    if n_gt == 0 or not dets:
        return 0.0
    dets = sorted(dets, key=lambda d: -d[0])
    matched = {}
    tp = np.zeros(len(dets))
    fp = np.zeros(len(dets))
    for k, (_, i, payload) in enumerate(dets):
        gts = iou_fn.gt[i]
        if len(gts) == 0:
            fp[k] = 1
            continue
        if i not in matched:
            matched[i] = np.zeros(len(gts), dtype=bool)
        ious = iou_fn(payload, gts)
        j = int(np.argmax(ious))
        if ious[j] >= 0.5 and not matched[i][j]:
            tp[k] = 1
            matched[i][j] = True
        else:
            fp[k] = 1
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_box_ap50(all_preds, all_gts):
    """all_preds: per image [N,5] (x1,y1,x2,y2,score); all_gts: per image
    [M,4]. Returns box AP@0.5."""
    n_gt = sum(len(g) for g in all_gts)
    dets = []
    for i, p in enumerate(all_preds):
        for row in np.asarray(p, dtype=np.float64).reshape(-1, 5):
            dets.append((row[4], i, row[:4]))

    def iou_fn(box, gts):
        return box_iou_one_to_many(box, np.asarray(gts, dtype=np.float64).reshape(-1, 4))
    iou_fn.gt = all_gts
    return _ap_from_matches(dets, n_gt, iou_fn)


def compute_mask_ap50(all_pred_masks, all_pred_scores, all_gt_masks):
    """Per image: pred masks [N,P,P] bool + scores [N]; GT masks [M,P,P]
    bool. Returns mask AP@0.5 (IoU measured at PROTO_SIZE resolution)."""
    n_gt = sum(len(g) for g in all_gt_masks)
    dets = []
    for i, (masks, scores) in enumerate(zip(all_pred_masks, all_pred_scores)):
        for n in range(len(masks)):
            dets.append((float(scores[n]), i, np.asarray(masks[n], dtype=bool)))

    def iou_fn(mask, gts):
        return mask_iou_one_to_many(mask, np.asarray(gts, dtype=bool))
    iou_fn.gt = all_gt_masks
    return _ap_from_matches(dets, n_gt, iou_fn)


@torch.no_grad()
def evaluate_maps(model, dataset, device, limit=None, conf_thres=0.05,
                  nms_thres=0.5, batch=8):
    """Run the model over a Dataset (items: image tensor [3,640,640] in
    [0,1], boxes [M,4] xyxy canvas coords, masks [M,160,160] float) and
    return (box_mAP50, mask_mAP50), class-agnostic."""
    model.eval()
    idxs = list(range(len(dataset)))
    if limit:
        idxs = idxs[:limit]
    all_pred_boxes, all_gt_boxes = [], []
    all_pred_masks, all_pred_scores, all_gt_masks = [], [], []
    for s in range(0, len(idxs), batch):
        chunk = idxs[s:s + batch]
        imgs = torch.stack([dataset[i][0] for i in chunk]).to(device)
        outs, protos = model(imgs)
        decoded = decode(outs, protos, img_size=IMG_SIZE, conf_thres=conf_thres)
        for j, i in enumerate(chunk):
            d = decoded[j]
            boxes = d["boxes"].cpu().numpy().reshape(-1, 4)
            scores = d["scores"].cpu().numpy()
            masks = d["masks"].cpu().numpy()
            if len(boxes):
                keep = nms(boxes, scores, nms_thres)
                boxes, scores, masks = boxes[keep], scores[keep], masks[keep]
            pred5 = np.concatenate(
                [boxes, scores[:, None]], axis=1) if len(boxes) else np.zeros((0, 5))
            all_pred_boxes.append(pred5)
            all_gt_boxes.append(dataset[i][1].numpy().reshape(-1, 4))
            all_pred_masks.append(masks)
            all_pred_scores.append(scores)
            all_gt_masks.append(dataset[i][2].numpy() >= 0.5)
    box_ap = compute_box_ap50(all_pred_boxes, all_gt_boxes)
    mask_ap = compute_mask_ap50(all_pred_masks, all_pred_scores, all_gt_masks)
    return box_ap, mask_ap
