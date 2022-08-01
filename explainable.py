import json
import joblib
import numpy as np

NUM_TASK = 6


def create_nodeIndex_dict(xgb_json):
    """ Create dictionary for finding the corresponding node_index and cross feature. """

    xgb_dict = {0: []}

    def add_cross_feat(xgb_json):
        """ Recursive adding decision rule, and using list to save the hierarchical tree decision rules. """
        
        # end of cross feature, e.g. {'nodeid': 65, 'leaf': 0.137037039} len = 2
        if len(xgb_json) < 8:  
            pass
        else:
            if xgb_dict[xgb_json['nodeid']] is None:
                xgb_dict[xgb_json['nodeid']] = []
            xgb_dict[xgb_json['yes']] = xgb_dict[xgb_json['nodeid']] + \
                [xgb_json['split'] + ' < ' + str(xgb_json['split_condition'])]
            xgb_dict[xgb_json['no']] = xgb_dict[xgb_json['nodeid']] + \
                [xgb_json['split'] + ' >= ' + str(xgb_json['split_condition'])]

            # recursive adding cross feature
            add_cross_feat(xgb_json['children'][0])
            add_cross_feat(xgb_json['children'][1])

    add_cross_feat(xgb_json)

    # Sort dictionary by keys
    xgb_dict = {k: xgb_dict[k] for k in sorted(xgb_dict.keys())}

    return xgb_dict


def transform_node_dict(xgb_json, leaf_dict):
    """ Transform the original tree node_idex to sequence node_index """

    sequence_node_dict = {}
    trees_node_dict = []
    for i in range(len(leaf_dict)):
        tmp_node_dict = create_nodeIndex_dict(xgb_json[i])
        trees_node_dict.append(tmp_node_dict)
        for k in tmp_node_dict.keys():
            try:
                sequence_node_dict[leaf_dict[i][k]] = tmp_node_dict[k]
            except:  # some node indexes are not used
                pass

    return sequence_node_dict, trees_node_dict


def get_node_dicts(model_path='Model/'):
    """ Load xgboost's booster json file, and transform to dictionary -> {node_index: cross_feature} """

    bank_leaf_dict = joblib.load(model_path + 'bank_leaf_dict')
    sino_leaf_dict = joblib.load(model_path + 'sino_leaf_dict')

    bank_xgb_json = json.load(open(model_path + 'bank_xgb_model.json'))
    sino_xgb_jsons = []
    for i in range(NUM_TASK):
        sino_xgb_jsons.append(json.load(open(model_path + 'sino_xgb_model_' + str(i) + '.json')))

    bank_node_dict, bank_trees_node_dict = transform_node_dict(bank_xgb_json, bank_leaf_dict)
    sino_node_dicts = []
    sino_trees_node_dicts = []
    for i in range(NUM_TASK):
        sino_node_dict, sino_trees_node_dict = transform_node_dict(sino_xgb_jsons[i], sino_leaf_dict[i])
        sino_node_dicts.append(sino_node_dict)
        sino_trees_node_dicts.append(sino_trees_node_dict)

    return bank_node_dict, bank_trees_node_dict, sino_node_dicts, sino_trees_node_dicts


def simplify_crossfeat(all_crossfeat):
    """ Get the min & max decision ranges of each features """
    crossfeat_range = {}
    for rule in all_crossfeat:
        feat_name, expression, value = rule.split(' ')
        value = float(value)

        if feat_name not in crossfeat_range.keys():
            # decision range: (min value, max value)
            crossfeat_range[feat_name] = (0, np.inf)

        if expression == '>=':
            if value > crossfeat_range[feat_name][0]:
                crossfeat_range[feat_name] = (value, crossfeat_range[feat_name][1])
        else:
            if value < crossfeat_range[feat_name][1]:
                crossfeat_range[feat_name] = (crossfeat_range[feat_name][0], value)

    return crossfeat_range


def get_crossfeat_by_group_nodeid(bank_crossfeat_dict, sino_crossfeat_dicts, X_bank, X_sino, group_node_id, simplify=True):
    """ Get non-repetitive group_node's cross feature """

    all_crossfeat = []
    if type(group_node_id) == list:
        for d in group_node_id:
            bank_crossfeat = bank_crossfeat_dict[X_bank[d]]
            all_crossfeat += bank_crossfeat
    else:
        bank_crossfeat = bank_crossfeat_dict[X_bank[group_node_id]]
        all_crossfeat += bank_crossfeat

    for task in range(NUM_TASK):
        if type(group_node_id) == list:
            for d in group_node_id:
                sino_crossfeat = sino_crossfeat_dicts[task][X_sino[task][d]]
                all_crossfeat += sino_crossfeat
        else:
            sino_crossfeat = sino_crossfeat_dicts[task][X_sino[task][group_node_id]]
            all_crossfeat += sino_crossfeat

    all_crossfeat = list(set(all_crossfeat))

    if simplify:
        all_crossfeat = simplify_crossfeat(all_crossfeat)

    return all_crossfeat


def count_feat(all_crossfeat):
    """ Count the frequency of decision features """
    feat_count = {}
    for rule in all_crossfeat:
        feat_name, expression, value = rule.split(' ')
        if feat_name not in feat_count.keys():
            feat_count[feat_name] = 0
        feat_count[feat_name] += 1

    feat_count = {k: v for k, v in sorted(feat_count.items(), key=lambda item: -item[1])}

    return feat_count


def get_crossfeat_by_group_treeid(bank_tree_dict, sino_tree_dicts, group_tree_id, feat_count=True):
    """ Get non-repetitive group_tree's cross feature """

    all_crossfeat = []
    bank_crossfeat = list(bank_tree_dict[group_tree_id].values())  # list of sub-trees
    # sum up all of the sub-tre cross features to a single list
    bank_crossfeat = sum(bank_crossfeat, [])
    all_crossfeat += bank_crossfeat

    for task in range(NUM_TASK):
        sino_crossfeat = list(sino_tree_dicts[task][group_tree_id].values())
        sino_crossfeat = sum(sino_crossfeat, [])
        all_crossfeat += sino_crossfeat

    if feat_count:
        return count_feat(all_crossfeat)
    else:
        return list(set(all_crossfeat))