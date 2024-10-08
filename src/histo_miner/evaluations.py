import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import numpy as np
import yaml
from attrdictionary import AttrDict as attributedict
from seaborn import heatmap


######################
### METRICS
######################

def get_fast_pq(true, pred, match_iou=0.5):
    """
	From Hovernet paper code.
	https://github.com/vqdang/hover_net/blob/master/metrics/stats_utils.py


    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing. 

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.    
    
    Fast computation requires instance IDs are in contiguous orderding 
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"

    ### For pairing we can also use the pairing function but for debug
    # and ocnsistancy we keep here the original scipt
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]



######################
### UTILS
######################

def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
        maxpred = np.max(new_pred) # for debug purposes 
    return new_pred



def pairing_cells(true, pred, match_iou=0.5):
    """
    Docstrings to fill later and even the function more
    """
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_true = list(paired_pred[paired_iou > match_iou] + 1)

    return paired_true, paired_pred


######################
### PLOTS
######################


def plot_conf_matrix(conf_matrix: np.ndarray, 
                     conf_matrix_normalized: np.ndarray,
                     conf_matrix_normalized_algpred: np.ndarray,
                     savefolder: str) -> None:
    """
    Plot nice confusion matrix with the help of seaborn (heatmap function).

    Parameters
    ----------
    Returns
    -------
    """
    # Store the name of classes
    selectedclass = [1, 2, 3, 4, 5]

    # Generate first plot, the classic confusion matrix with no normalization
    _, ax = plt.subplots(figsize=(9, 6))
    heatmap(conf_matrix, annot=True, linewidths=.5, ax=ax, fmt='g', cmap='YlGnBu')
    plt.title('Confusion Matrix ')
    plt.ylabel('Groundtruth')
    ax.set_xlabel('Predicted')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    my_xticks = ['granul        ', 'lympho         ', 'plasma       ', 'stroma       ', 'tumor      ']
    my_yticks = ['      granul', '      lympho', '      plasma', '      stroma', '       tumor']
    plt.xticks(ticks=selectedclass, labels=my_xticks)
    plt.yticks(ticks=selectedclass, labels=my_yticks)
    plt.savefig(savefolder + '/' + 'conf_mat.png', dpi=1000, bbox_inches='tight')
    #plt.show()

    # Generate second plot, the confusion matrix with Recall normalization (more conventional)
    _, ax = plt.subplots(figsize=(9, 6))
    heatmap(conf_matrix_normalized, annot=True, linewidths=.5, ax=ax, fmt='.2g', cmap='YlGnBu')
    plt.title('Confusion Matrix: Recall Normalization')
    plt.ylabel('Groundtruth')
    ax.set_xlabel('Predicted')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    my_xticks = ['granul        ', 'lympho         ', 'plasma       ', 'stroma       ', 'tumor      ']
    my_yticks = ['      granul', '      lympho', '      plasma', '      stroma', '       tumor']
    plt.xticks(ticks=selectedclass, labels=my_xticks)
    plt.yticks(ticks=selectedclass, labels=my_yticks)
    plt.savefig(savefolder + '/' + 'conf_mat_truenorm.png', dpi=1000, bbox_inches='tight')
    #plt.show()
    
    # Generate third plot, the confusion matrix with Prediction normalization
    _, ax = plt.subplots(figsize=(9, 6))
    heatmap(conf_matrix_normalized_algpred, annot=True, linewidths=0.5, ax=ax, fmt='.2g', cmap='YlGnBu')
    plt.title('Confusion Matrix: Prediction Normalization ')
    plt.ylabel('Groundtruth')
    ax.set_xlabel('Predicted')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    my_xticks = ['granul        ', 'lympho      ', 'plasma      ', 'stroma       ', 'tumor      ']
    my_yticks = ['       granul', '      lympho', '      plasma', '      stroma', '       tumor']
    plt.xticks(ticks=selectedclass, labels=my_xticks)
    plt.yticks(ticks=selectedclass, labels=my_yticks)
    plt.savefig(savefolder + '/' + 'conf_mat_prednorm.png', dpi=1000, bbox_inches='tight')
    #plt.show()
    #print('All plot shown')








