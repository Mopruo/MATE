import copy
import joblib
import torch
import numpy as np
import xgboost as xgb

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from .evaluation import FE_evaluate, MATE_evaluation
from .models.FE import FE
from .models.MATE import MATE
from .tree_func import process_leaf_idx, get_bank_xgb_leaves, get_sino_xgb_leaves
from .variables import SINO_START_INDEX


def pretrain_bank_xgb(data, n_estimators=15, test_size=0.2, random_state=3, save_model=True, model_path='Model/'):
    X_bank, X_sino, y_bank, y_sino, y_sino_cls, y_sino_xgbcls, sino_comp = data
    
    X_bank = X_bank[:SINO_START_INDEX]  # exclude sino bank branches

    X_train_bank, X_test_bank, y_train_bank, y_test_bank = train_test_split(
        X_bank, y_bank, test_size=test_size, random_state=random_state)

    bank_xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',  # squarederror, pseudohubererror
        subsample=0.85,
        colsample_bytree=0.7,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=n_estimators,
        enable_experimental_json_serialization=True,
    )

    # Use part of the training data to train the tree model (optional)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_bank, y_train_bank.bank_count.values, train_size=0.7, random_state=42)

    bank_xgb_model.fit(X_train, y_train.reshape(-1,))

    print('* pretrain_bank_xgb: ')
    print('train mse: ', mean_squared_error(y_train_bank.bank_count.values, bank_xgb_model.predict(X_train_bank)))
    print('test mse: ', mean_squared_error(y_test_bank.bank_count.values, bank_xgb_model.predict(X_test_bank)))
    print('-'*50)

    if save_model:
        X_leaves = bank_xgb_model.apply(X_bank)
        all_leaves, bank_max_leaf, record_leaf_dict, bank_leaf_dict = process_leaf_idx(X_leaves)

        bank_xgb_model.get_booster().dump_model(model_path + 'bank_xgb_model.json', with_stats=False, dump_format='json')
        joblib.dump(bank_xgb_model, model_path + 'bank_xgb_model.model')
        joblib.dump(bank_leaf_dict, model_path + 'bank_leaf_dict')
        joblib.dump(bank_max_leaf, model_path + 'bank_max_leaf')

    return bank_xgb_model, (X_train_bank, X_test_bank, y_train_bank, y_test_bank)


def pretrain_FE(X_train, X_test, y_train, y_test, leaf_dim=20, epochs=10, model_path='Model/', device="cuda:0"):
    bank_xgb_model = joblib.load(model_path + 'bank_xgb_model.model')
    bank_max_leaf = joblib.load(model_path + 'bank_max_leaf')

    # Transform data to xgboost's leaves
    X_train_leaves = get_bank_xgb_leaves(X_train, model_path=model_path)
    X_test_leaves = get_bank_xgb_leaves(X_test, model_path=model_path)

    y_train_cls, y_train_reg = y_train.bank_binary.values, y_train.bank_count.values
    y_test_cls, y_test_reg = y_test.bank_binary.values, y_test.bank_count.values

    FE_model = FE(max_leaf=bank_max_leaf,
                  leaf_dim=leaf_dim,
                  n_estimators=bank_xgb_model.n_estimators,
                  hidden_dim=12)
    FE_model = FE_model.to(device)
    
    FE_model.fit(X=X_train_leaves, y=(y_train_cls, y_train_reg),
                 validation_data=(X_test_leaves, y_test_cls, y_test_reg), batch_size=128, epochs=epochs)

    # Freeze FE model's parameters for downstream task training
    for p in FE_model.parameters():
        p.requires_grad = False

    FE_evaluate(FE_model, X_test_leaves, y_test)

    torch.save(FE_model, model_path + 'FE_model.pt')

    return FE_model


def pretrain_sino_xgb(X, y, n_estimators=15, save_model=True, model_path='Model/'):
    sino_xgb_models = [
        xgb.XGBRegressor(
            objective="reg:squarederror",  # pseudohubererror squarederror
            subsample=1.,
            colsample_bytree=0.6,
            max_depth=5,
            learning_rate=0.07,
            n_estimators=n_estimators,
            enable_experimental_json_serialization=True,
        )
        for i in range(6)
    ]

    for i in range(6):
        sino_xgb_models[i].fit(X, y[:, i].reshape(-1,))

    if save_model:
        sino_leaf_dict = []
        sino_max_leaf = []
        for i in range(len(sino_xgb_models)):  # num tasks
            X_leaves = sino_xgb_models[i].apply(X)
            all_leaves, max_leaves, new_leaf_index, leaf_dict_list = process_leaf_idx(X_leaves)
            sino_leaf_dict.append(leaf_dict_list)
            sino_max_leaf.append(max_leaves)

            sino_xgb_models[i].get_booster().dump_model(model_path + 'sino_xgb_model_' +
                                                        str(i) + '.json', with_stats=False, dump_format='json')
            joblib.dump(sino_xgb_models[i], model_path + 'sino_xgb_model_' + str(i) + '.model')
        joblib.dump(sino_leaf_dict, model_path + 'sino_leaf_dict')
        joblib.dump(sino_max_leaf, model_path + 'sino_max_leaf')

    return sino_xgb_models


def training_MATE(
    X_data,
    y_data,
    FE_model,
    n_estimators=15,
    num_class=5,
    head_num=2,
    use_FE=True,
    use_self_att=True,
    prob_loss=True,
    comp_attention=True,
    save_best=True,
    model_path='Model/',
    batch_size=16,
    epochs=50,
    device='cuda:0',
):
    
    if len(X_data) == 3:
        X_train_bank, X_train_sino, X_train_comp = X_data
        y_train, y_train_reg = y_data
        validation_data = None
    else:
        X_train_bank, X_test_bank, X_train_sino, X_test_sino, X_train_comp, X_test_comp = X_data
        y_train, y_train_reg, y_test = y_data
    
    # training xgboost models for every task
    sino_xgb_models = pretrain_sino_xgb(X_train_sino, y_train_reg, n_estimators=n_estimators,
                                        save_model=True, model_path=model_path)
    sino_max_leaf = joblib.load(model_path + 'sino_max_leaf')
    
    # transform numerical data to xgboost leaf id
    X_train_bank = get_bank_xgb_leaves(X_train_bank, model_path=model_path)
    X_train_sino = get_sino_xgb_leaves(X_train_sino, model_path=model_path)
    if len(X_data) != 3:
        X_test_bank = get_bank_xgb_leaves(X_test_bank, model_path=model_path)
        X_test_sino = get_sino_xgb_leaves(X_test_sino, model_path=model_path)
        validation_data = ((X_test_bank, X_test_sino, X_test_comp), y_test)
        
    # Using traditional cross-entropy
    if not prob_loss:
        y_train = np.argmax(y_train, axis=-1)
        if validation_data is not None:
            y_test = np.argmax(y_test, axis=-1)

    MATE_model = MATE(
        FE_model=FE_model,
        sino_estimators=n_estimators,
        max_leaf=sino_max_leaf,
        num_task=6,
        num_class=num_class,
        head_num=head_num,
        use_FE=use_FE,
        use_self_att=use_self_att,
        prob_loss=prob_loss,
        comp_attention=comp_attention,
        device=device,
    )

    MATE_model = MATE_model.to(device)
    MATE_model.fit(
        X=(X_train_bank, X_train_sino, X_train_comp),
        y=y_train,
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs,
        save_best=save_best,
        model_path=model_path,
    )

    return MATE_model


def Kfold_MATE_validation(
    data,
    FE_model,
    K=5,
    random_state=1,
    n_estimators=15,
    num_class=7,
    head_num=4,
    batch_size=16,
    epochs=70,
    save_best=True,
    model_path='Model/',
    use_FE=False,
    use_self_att=False,
    prob_loss=True,
    comp_attention=False,
    device='cuda:0',
):
    X_bank, X_sino, y_bank, y_sino_reg, y_sino_cls, y_sino_xgbcls, sino_comp = data
    X_bank = X_bank.loc[SINO_START_INDEX:].reset_index(drop=True)

    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)

    acc_list, mae_list, kendalltau_list, weightedtau_list = [], [], [], []
    task_acc_list, task_mae_list = [], []
    hit_rate = np.array([0., 0., 0., 0., 0., 0., 0.])
    for fold, (train_index, test_index) in enumerate(kf.split(X_bank)):
        X_train_bank, X_test_bank = X_bank.reindex(train_index), X_bank.reindex(test_index)
        X_train_sino, X_test_sino = X_sino.reindex(train_index), X_sino.reindex(test_index)
        y_train_comp, y_test_comp = sino_comp[train_index], sino_comp[test_index]
        y_train_reg, y_test_reg = y_sino_reg[train_index], y_sino_reg[test_index]
        y_train_cls, y_test_cls = y_sino_cls[train_index], y_sino_cls[test_index]
        y_train_xgbcls, y_test_xgbcls = y_sino_xgbcls[train_index], y_sino_xgbcls[test_index]
        
        FE_model_ = copy.deepcopy(FE_model)
        MATE_model = training_MATE(
            X_data=(X_train_bank, X_test_bank, X_train_sino, X_test_sino, y_train_comp, y_test_comp),
            y_data=(y_train_cls, y_train_reg, y_test_cls),
            FE_model=FE_model_,
            n_estimators=n_estimators,
            num_class=num_class,
            head_num=head_num,
            batch_size=batch_size,
            epochs=epochs,
            save_best=save_best,
            model_path=model_path,
            use_FE=use_FE,
            use_self_att=use_self_att,
            prob_loss=prob_loss,
            comp_attention=comp_attention,
            device=device,
        )

        print("-" * 50)
        print("* Fold: ", fold)
        MATE_model = torch.load(model_path + 'MATE_model.pt')
        X_test_bank = get_bank_xgb_leaves(X_test_bank, model_path=model_path)
        X_test_sino = get_sino_xgb_leaves(X_test_sino, model_path=model_path)
        top_k_acc, kendalltau, weightedtau, acc, mae, task_acc, task_mae = MATE_evaluation(
            MATE_model, X_test_bank, X_test_sino, y_test_comp, y_test_reg, y_test_cls,)

        print("-" * 50)
        hit_rate += np.array(top_k_acc)
        acc_list.append(acc)
        mae_list.append(mae)
        kendalltau_list.append(kendalltau)
        weightedtau_list.append(weightedtau)
        task_acc_list.append(task_acc)
        task_mae_list.append(task_mae)

    print("-" * 50)
    print("K Fold Average: ")
    print("Hit Rate: ", hit_rate / K)
    print(
        "ACC: {:.4f} | MAE: {:.4f} | Tau: {:.4f} | Weight_Tau: {:.4f}".format(
            np.mean(acc_list),
            np.mean(mae_list),
            np.mean(kendalltau_list),
            np.mean(weightedtau_list),
        )
    )
    
    return np.mean(acc_list), np.mean(mae_list), np.mean(task_acc_list, axis=0), np.mean(task_mae_list, axis=0)
