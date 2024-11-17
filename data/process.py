import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def get_quantile_locations(x, args):
    num_quantile = 5
    p = np.linspace(0, 1, num_quantile+1)
    quantile = np.quantile(x, p).astype('float32')
    quantile.sort()
    args.locs = quantile


def get_tree_locs_and_scales(x, args):
    split_loc = []
    train_col = x
    clf = DecisionTreeClassifier(max_leaf_nodes=8,
                                    min_samples_leaf=16,
                                    min_impurity_decrease=0.0)
    tree = clf.fit(train_col.reshape(-1, self.series_size), self.targets).tree_
    for i in range(tree.node_count):
        if tree.children_left[i] != tree.children_right[i]:
            split_loc.append(tree.threshold[i])
    split_loc.append(train_col.min())
    split_loc.append(train_col.max())
    split_loc = np.array(split_loc).astype('float32')
    split_loc.sort()
    if args.use_tree_locs:
        args.locs = split_loc
    if args.use_tree_scales:
        scales = np.sort(split_loc[1:] - split_loc[:-1])
        max_scale = abs((scales - tree.threshold[0]).max())
        pow_max = int(math.log(max(1e-7, max_scale), args.base_bias)) + 2
        pow_min = int(math.log(max(1e-7, scales.min()), args.base_bias)) - 2
        pow_interval = (pow_max - pow_min) / (args.num_bias - 1)
        args.scales = [args.base_bias ** (pow_min + pow_interval * p) for p in range(args.num_bias)]
