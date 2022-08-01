import joblib
import numpy as np


def replace_with_dict(ar, dic):
    """
    Parameters
    --------------------
    ar: original leaves in a single tree
    dic: a dict that record the corresponding unique leaf id
    
    Returns
    --------------------
    transformed unique leaves id
    """
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    # Drop the magic bomb with searchsorted to get the corresponding
    # places for a in keys (using sorter since a is not necessarily sorted).
    # Then trace it back to original order with indexing into sidx
    # Finally index into values for desired output.
    return v[sidx[np.searchsorted(k, ar, sorter=sidx)]]


def transform_leaf_idx(leaves, leaf_dict_list):
    " transform all the leaves to unique leaf_id by leaf_dict_list "
    for i in range(leaves.shape[1]):
        leaves[:, i] = replace_with_dict(leaves[:, i], leaf_dict_list[i])
    return leaves


def process_leaf_idx(X_leaves):
    """
    Returns:
    --------------------
    leaves: X_leaves after transform (unique id for each leaf node)
    total_leaves: maximum leaf id
    record_leaf_dict: a dictionary that record e.g. {uniq_id: {tree_id: original_node_id}}
    leaf_dict_list: a list of dicts (length=n_estimators), 
                    each dict record e.g. {original_node_id: uniq_id, original_node_id: uniq_id ...}
    """
    leaves = X_leaves.copy()

    record_leaf_dict = dict()  # Use dictionary to record leaf index
    leaf_dict_list = []
    total_leaves = 0

    for t in range(X_leaves.shape[1]):  # iterate for each column (tree)
        column = X_leaves[:, t]
        unique_vals = list(sorted(set(column)))
        new_idx = {v: (i + total_leaves) for i, v in enumerate(unique_vals)}
        leaf_dict = dict()
        for i, v in enumerate(unique_vals):
            leaf_id = i + total_leaves
            record_leaf_dict[leaf_id] = {t: v}  # record the original tree index and leaf index
            leaf_dict[v] = leaf_id
        leaf_dict_list.append(leaf_dict)
        leaves[:, t] = [new_idx[v] for v in column]  # transform the original leaf index to unique sequential value
        total_leaves += len(unique_vals)  # cumcount the number of leaves

    assert leaves.ravel().max() == total_leaves - 1
    return leaves, total_leaves, record_leaf_dict, leaf_dict_list


def get_bank_xgb_leaves(X_bank, model_path='Model/'):
    """ transform original feature to pre-trained xgboost leaves id """
    
    bank_leaf_dict = joblib.load(model_path + 'bank_leaf_dict')
    xgb_model = joblib.load(model_path + 'bank_xgb_model.model')

    X_leaves = xgb_model.apply(X_bank)
    X_leaves = transform_leaf_idx(X_leaves, bank_leaf_dict)

    return X_leaves


def get_sino_xgb_leaves(X_sino, model_path='Model/'):
    """ transform original feature to multiple pre-trained xgboost leaves id """
    
    sino_leaf_dict = joblib.load(model_path + 'sino_leaf_dict')
    
    for i in range(len(sino_leaf_dict)):
        xgb_model = joblib.load(model_path + 'sino_xgb_model_' + str(i) + '.model')
        X_leaves = xgb_model.apply(X_sino)
        X_leaves = np.expand_dims(transform_leaf_idx(X_leaves, sino_leaf_dict[i]), axis=1)
        if i == 0:
            all_X_leaves = X_leaves
        else:
            all_X_leaves = np.concatenate((all_X_leaves, X_leaves), axis=1)
            
    return all_X_leaves


