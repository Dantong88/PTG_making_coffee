import logging
from sklearn.metrics import auc, precision_recall_curve, recall_score, f1_score, average_precision_score
import numpy as np

log = logging.getLogger("ptg_eval")


class EvalMetrics():
    def __init__(self, labels, gt_true_mask, dets_per_valid_time_w, output_fn='metrics/txt'):
        """
        :param labels: Array of class labels (str)
        :param gt_true_pos_mask: Matrix of size (number of valid time windows x number classes) where True
            indicates a true class example, False inidcates a false class example
        :param dets_per_valid_time_w: Matrix of size (number of valid time windows x number classes) filled with 
            the max confidence score per class for any detections in the time window
        :param output_fn: Path (str) to a file
        """
        self.labels = labels
        self.gt_true_mask = gt_true_mask
        self.dets_per_valid_time_w = dets_per_valid_time_w

        self.output_fn = output_fn

    def precision(self):
        with open(self.output_fn, "w") as f:
            for id, label in enumerate(self.labels):
                class_dets_per_time_w = self.dets_per_valid_time_w[:, id]
                mask_per_class = self.gt_true_mask[:, id]

                tp = class_dets_per_time_w[mask_per_class]
                fp = class_dets_per_time_w[~mask_per_class]

                s = np.hstack([tp, fp]).T
                y_true = np.hstack([np.ones(len(tp), dtype=bool),
                        np.zeros(len(fp), dtype=bool)]).T
                s.shape = (-1, 1)
                y_true.shape = (-1, 1)

                #precision, recall, thresholds = precision_recall_curve(y_true, s)
                precision = average_precision_score(y_true, s)
                # TODO add recall

                

                f.write(f'{self.labels[id]}: {precision}\n')
