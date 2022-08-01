import math
import numpy as np

from scipy import stats
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


def acc_eval(preds, labels):
    sorted_preds_id = np.argsort(preds, axis=0)
    sorted_labels_id = np.argsort(labels, axis=0)
    acc_lists = []
    # top N% accuracy
    for N in [0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.7]:
        num_data = math.ceil(preds.shape[0] * N)
        top_preds = sorted_preds_id[ : num_data]
        top_labels = sorted_labels_id[ : num_data]
        Intersection = np.intersect1d(top_preds,top_labels)
        acc = Intersection.shape[0] / num_data
        acc_lists.append(acc)
    
    return np.array(acc_lists)


def FE_evaluate(model, X_test, y_test):
    model.eval()
    pred_cls, pred_reg = model.predict(X_test)

    y_test_copy = y_test.copy()
    y_test_copy['preds_cls'] = pred_cls
    y_test_copy['preds_reg'] = pred_reg

    print('* FE_model pretrain: ')
    print('test all acc: ', accuracy_score(y_test_copy.bank_binary.values, y_test_copy.preds_cls.values))
    print('test bank==0 : ', accuracy_score(
        y_test_copy.loc[y_test_copy.bank_binary == 0].bank_binary, y_test_copy.loc[y_test_copy.bank_binary == 0].preds_cls))
    print('test bank==1 : ', accuracy_score(
        y_test_copy.loc[y_test_copy.bank_binary == 1].bank_binary, y_test_copy.loc[y_test_copy.bank_binary == 1].preds_cls))
    print('-'*50)


def MATE_evaluation(model, X_bank, X_sino, X_comp, y_reg, y_cls):
    total_preds = np.zeros(y_reg.shape[0])
    total_labels = np.zeros(y_reg.shape[0])
    preds = model.predict(X_bank, X_sino, X_comp)
    
    # Get classification result
    preds = np.argmax(preds, axis=-1)
    y_cls = np.argmax(y_cls, axis=-1)
    
    weights = [1, 1, 2, 2, 2, 2]
    sum_acc, sum_mae = 0, 0
    task_acc = []
    task_mae = []
    for i in range(6): 
        acc = accuracy_score(y_cls[:, i], preds[:, i])
        cls_mae = (np.absolute( preds[:, i].reshape(-1) - y_cls[:, i] )).mean()
        task_acc.append(acc)
        task_mae.append(cls_mae)
        sum_acc += acc
        sum_mae += cls_mae
        print('Task %d ACC:%3.4f | MAE:%3.4f'%(i, acc, cls_mae))
        total_preds += preds[:, i].reshape(-1) * weights[i]
        total_labels += y_cls[:, i].reshape(-1) * weights[i]
    
    weightedtau = stats.weightedtau(total_preds, total_labels).correlation
    kendalltau = stats.kendalltau(total_preds, total_labels).correlation
    acc_lists = acc_eval(total_preds, total_labels)
    print('* Avg acc: %2.4f  |  Avg mae: %2.4f'%(sum_acc/6., sum_mae/6.))
    print('* Total Top K% Accuracy:', acc_lists)
    print('* Total Kendalltau:', kendalltau)
    print('* Total Weightedtau:', weightedtau)
    
    return acc_lists, kendalltau, weightedtau, sum_acc/6., sum_mae/6., task_acc, task_mae