#heatmap history
import numpy as np


class heatmap_history():
    def __init__(self):
        # was the line detected in the last iteration?
        self.heatmaps = []

    def combined_heatmap(self, heatmap):
        self.heatmaps.append(heatmap)
        heatmap_count = len(self.heatmaps)
        heatmaps = []
        if (heatmap_count < 9):
            heatmaps = self.heatmaps
        else:
            heatmaps = self.heatmaps[(heatmap_count-9):]

        final_heatmap = np.zeros_like(heatmap)
        for hm in heatmaps:
            final_heatmap[hm == 1] += 1
            final_heatmap[hm == 2] += 1
            final_heatmap[hm == 3] += 1
            final_heatmap[hm == 4] += 1
            final_heatmap[hm == 5] += 1
            final_heatmap[hm == 6] += 1
            final_heatmap[hm == 7] += 1
            final_heatmap[hm == 8] += 1
            final_heatmap[hm == 9] += 1
            final_heatmap[hm == 10] += 1
            final_heatmap[hm == 11] += 1
            final_heatmap[hm == 12] += 1
            final_heatmap[hm == 13] += 1
            final_heatmap[hm == 14] += 1
            final_heatmap[hm == 15] += 1
            final_heatmap[hm == 16] += 1
            final_heatmap[hm == 17] += 1
            final_heatmap[hm == 18] += 1
            final_heatmap[hm == 19] += 1
            final_heatmap[hm == 20] += 1
            final_heatmap[hm == 21] += 1
            final_heatmap[hm == 22] += 1


        return final_heatmap
