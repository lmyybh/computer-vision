import torch


class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, low_threshold, high_threshold, allow_low_quality_matches=False):
        assert low_threshold <= high_threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        assert match_quality_matrix.numel() > 0 and match_quality_matrix.shape[0] > 0

        # 找到和每个 predicted_box 匹配最好的 gt_box, 即每行最大 iou
        vals, matched_idxs = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matched_idxs.clone()

        # 与阈值比较，用负数标记低于 high_threshold 的 box
        below = vals < self.low_threshold
        between = (vals >= self.low_threshold) & (vals < self.high_threshold)
        matched_idxs[below] = Matcher.BELOW_LOW_THRESHOLD
        matched_idxs[between] = Matcher.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            matched_idxs = self.set_low_quality_matches_(
                matched_idxs, all_matches, match_quality_matrix
            )

        return matched_idxs

    def set_low_quality_matches_(self, matched_idxs, all_matches, match_quality_matrix):
        vals, _ = match_quality_matrix.max(dim=1)
        indexs = torch.nonzero(match_quality_matrix == vals.reshape(-1, 1))[:, 1]
        matched_idxs[indexs] = all_matches[indexs]
        return matched_idxs
