#label history
import numpy as np


class labels_history():
    def __init__(self):
        # was the line detected in the last iteration?
        self.labels = []

    def averaged_labels(self, label):
        self.labels.append(label)
        label_count = len(self.labels)
        labels = []
        if (label_count < 5):
            labels = self.labels
        else:
            labels = self.labels[(label_count-5):]

        final_label = np.zeros_like(label)
        for lbl in labels:
            final_label[lbl >= 1] += 1

        return final_label
