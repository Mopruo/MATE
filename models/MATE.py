import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from ..optimizers.radam import RAdam
from sklearn.metrics import accuracy_score, mean_squared_error
from snorkel.classification import cross_entropy_with_probs


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))
    
    
class Sliding_conv_module(nn.Module):
    def __init__(self, kernel_size):
        super(Sliding_conv_module, self).__init__()
        self.kernel_size = kernel_size
        self.mish = Mish()
        self.conv_1 = nn.Conv1d(in_channels=1,
                                out_channels=int(self.kernel_size/2),
                                kernel_size=self.kernel_size, stride=self.kernel_size)

        self.conv_2 = nn.Conv1d(in_channels=1,
                                out_channels=int(self.kernel_size/2),
                                kernel_size=int(self.kernel_size/2), stride=int(self.kernel_size/2))

    def forward(self, X):
        X = X.view(X.shape[0], 1, -1)
        X = self.mish(self.conv_1(X))
        X = X.view(X.shape[0], 1, -1)
        X = self.conv_2(X)
        X = X.view(X.shape[0], -1, int(self.kernel_size/2))
        return X    
    

# Multi-task Attentive Tree-Enhanced Model
class MATE(nn.Module):
    def __init__(
        self,
        FE_model,
        sino_estimators,
        max_leaf,
        num_task=6,
        num_class=5,
        head_num=2,
        use_FE=True,
        use_self_att=True,
        prob_loss=False,
        comp_attention=True,
        device="cuda:0",
    ):
        super(MATE, self).__init__()
        self.FE_model = FE_model
        self.num_task = num_task
        self.num_class = num_class
        self.head_num = head_num
        self.max_leaf = max_leaf
        self.leaf_dim = FE_model.leaf_dim
        self.bank_estimators = FE_model.n_estimators
        self.sino_estimators = sino_estimators
        self.use_FE = use_FE
        self.use_self_att = use_self_att
        self.prob_loss = prob_loss
        self.comp_attention = comp_attention
        self.device = device
        self.mish = Mish()

        if self.use_FE:
            self.group_n = self.num_task + 1
        else:
            self.group_n = self.num_task

        comp_hidden_dim = 8
        self.leaf_embeddings = nn.ModuleList()
        self.comp_layer = nn.ModuleList()
        self.multi_att_layer = nn.ModuleList()
        self.comp_multi_att_layer = nn.ModuleList()
        self.layernorm_layer = nn.ModuleList()
        self.comp_layernorm_layer = nn.ModuleList()
        self.sliding_conv_layer = nn.ModuleList()
        self.final_layer = nn.ModuleList()

        for i in range(self.num_task):
            # Embedding layer for tree leaves
            leaf_embedding = nn.Embedding(self.max_leaf[i], self.leaf_dim)
            self.leaf_embeddings.append(leaf_embedding)

            # sliding_conv Layer for extracting features from a group of tree leaves embedding
            group_dim = self.leaf_dim * self.group_n
            sliding_conv = Sliding_conv_module(group_dim)
            self.sliding_conv_layer.append(sliding_conv)
            group_dim = int(group_dim / 2)  # dimension reduction after sliding_conv

            if self.use_self_att:
                multi_att_layer = nn.MultiheadAttention(
                    embed_dim=group_dim, num_heads=self.head_num, dropout=0.15
                )
                layernorm = nn.LayerNorm((self.sino_estimators, group_dim))
                self.multi_att_layer.append(multi_att_layer)
                self.layernorm_layer.append(layernorm)

                if self.comp_attention:  # Enable Self-Attention & Competitive Self-Attention
                    final_kernel_size = group_dim * 3 + comp_hidden_dim
                else:  # Enable Self-Attention, Disable Competitive Self-Attention
                    final_kernel_size = group_dim * 2
            else:
                if self.comp_attention:  # Disable Self-Attention, Enable Competitive Self-Attention
                    final_kernel_size = group_dim * 2 + comp_hidden_dim
                else:  # Disable Self-Attention & Competitive Self-Attention
                    final_kernel_size = group_dim

            if self.comp_attention:
                comp_layer = nn.Linear(10, comp_hidden_dim)
                comp_multi_att_layer = nn.MultiheadAttention(
                        embed_dim=group_dim + comp_hidden_dim, num_heads=self.head_num, dropout=0.15
                    )
                comp_layernorm = nn.LayerNorm((self.sino_estimators, group_dim + comp_hidden_dim))
                self.comp_layer.append(comp_layer)
                self.comp_multi_att_layer.append(comp_multi_att_layer)
                self.comp_layernorm_layer.append(comp_layernorm)

            final_layer = nn.Conv1d(in_channels=self.sino_estimators,
                                    out_channels=self.num_class * self.sino_estimators,
                                    kernel_size=final_kernel_size,
                                    groups=self.sino_estimators)
            self.final_layer.append(final_layer)

    def forward(self, X_bank, X_sino, X_comp):
        # get leaf embeddings of all tasks
        for i in range(self.num_task):
            sino_embedding = self.leaf_embeddings[i](X_sino[:, i])
            if i == 0:
                if self.use_FE:
                    cls_out, reg_out, bank_embedding = self.FE_model(X_bank)
                    X = torch.cat((bank_embedding, sino_embedding), 1)
                else:  # only use sino_xgb embeddings
                    X = sino_embedding

            else:
                X = torch.cat((X, sino_embedding), 1)
        
        if self.use_FE:
            X_mix_task = self.shuffle_channels(X, self.num_task + 1)
        else:
            X_mix_task = self.shuffle_channels(X, self.num_task)
        
        for task in range(self.num_task):
            sliding_out = self.sliding_conv_layer[task](X_mix_task)
            task_X = sliding_out
            
            if self.use_self_att:
                # permute to fit the input shape of torch.nn.multi_head_attention
                _sliding_out = sliding_out.permute(1, 0, 2)
                X_att, X_att_weights = self.multi_att_layer[task](_sliding_out, _sliding_out, _sliding_out)
                X_att = X_att.permute(1, 0, 2)
                # X_att = self.layernorm_layer[task](X_att)
                task_X = torch.cat((task_X, X_att), dim=-1)

            if self.comp_attention:
                # transform the competitive features, and concat to each group embedding (sliding_out)
                X_comp_reduce = self.comp_layer[task](X_comp)
                X_comp_task = torch.cat((X_comp_reduce.unsqueeze(1).expand(X_comp_reduce.size(
                        0), self.sino_estimators, X_comp_reduce.size(1)), sliding_out), dim=-1)

                _X_comp_task = X_comp_task.permute(1, 0, 2) 
                X_att_comp, X_att_comp_weights = self.comp_multi_att_layer[task](
                        _X_comp_task, _X_comp_task, _X_comp_task)
                X_att_comp = X_att_comp.permute(1, 0, 2) 
                task_X = torch.cat((task_X, X_att_comp), dim=-1)

            task_X = task_X.view(task_X.shape[0], self.sino_estimators, -1)
            # task_X = F.dropout(task_X, p=0.2, training=self.training)

            final_out = self.final_layer[task](task_X)
            final_out = final_out.view(final_out.shape[0], self.sino_estimators, self.num_class)
            final_mean_out = torch.mean(final_out, dim=1)

            if task == 0:
                if self.use_self_att:
                    all_X_att_weights = X_att_weights.unsqueeze(1)
                if self.comp_attention:
                    all_X_att_comp_weights = X_att_comp_weights.unsqueeze(1)
                final_outs = final_out.unsqueeze(1)
                final_mean_outs = final_mean_out.unsqueeze(1)
            else:
                if self.use_self_att:
                    all_X_att_weights = torch.cat((all_X_att_weights, X_att_weights.unsqueeze(1)), dim=1)
                if self.comp_attention:
                    all_X_att_comp_weights = torch.cat((all_X_att_comp_weights, X_att_comp_weights.unsqueeze(1)), dim=1)
                final_outs = torch.cat((final_outs, final_out.unsqueeze(1)), dim=1)
                final_mean_outs = torch.cat((final_mean_outs, final_mean_out.unsqueeze(1)), dim=1)

        pred_values = [final_mean_outs, final_outs]
        if self.use_self_att:
            pred_values.append(all_X_att_weights)
        if self.comp_attention:
            pred_values.append(all_X_att_comp_weights)

        return pred_values

    def shuffle_channels(self, x, groups):
        """shuffle channels of a 3-D Tensor"""
        batch_size, channels, feature = x.size()

        assert channels % groups == 0
        channels_per_group = channels // groups

        # split into groups
        x = x.view(batch_size, groups, channels_per_group, feature)

        x = x.transpose(1, 2).contiguous()

        # reshape back
        x = x.view(batch_size, channels, feature)
        return x
    
    def evaluate(self, X, y):
        assert len(X) == 3
        X_bank, X_sino, X_comp = X
        y_preds = self.predict(X_bank, X_sino, X_comp)

        acc_sum = 0
        for i in range(self.num_task):
            if self.prob_loss:
                acc_sum += accuracy_score(np.argmax(y[:, i], axis=-1), np.argmax(y_preds[:, i], axis=-1))
            else:
                acc_sum += accuracy_score(y[:, i], np.argmax(y_preds[:, i], axis=-1))
        acc_sum = acc_sum / self.num_task

        return acc_sum

    def predict(self, X_bank, X_sino, X_comp, return_all=False):
        model = self.eval()

        X_bank = torch.tensor(X_bank).long()
        X_sino = torch.tensor(X_sino).long()
        X_comp = torch.tensor(X_comp).float()

        test_dataset = Data.TensorDataset(X_bank, X_sino, X_comp)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False,)

        for step, (X_bank, X_sino, X_comp) in enumerate(test_loader):
            X_bank = X_bank.to(self.device)
            X_sino = X_sino.to(self.device)
            X_comp = X_comp.to(self.device)

            pred_values = model(X_bank, X_sino, X_comp)
            for i, v in enumerate(pred_values):
                pred_values[i] = pred_values[i].detach().cpu().numpy()

            pred_prob = pred_values[0]
            #preds_mean = np.argmax(preds_mean, axis=-1)

            if step == 0:
                y_pred = pred_prob
                all_pred_values = pred_values
            else:
                y_pred = np.concatenate((y_pred, pred_prob), axis=0)
                for i, v in enumerate(pred_values):
                    all_pred_values[i] = np.concatenate((all_pred_values[i], pred_values[i]), axis=0)

        if return_all:
            return all_pred_values
        else:
            return y_pred

    def fit(self, X, y, validation_data=None, batch_size=16, epochs=15, save_best=True, model_path='Model/'):

        assert len(X) == 3
        X_bank_original, X_sino_original, X_comp_original = X

        X_train_bank = torch.tensor(X_bank_original).long()
        X_train_sino = torch.tensor(X_sino_original).long()
        X_train_comp = torch.tensor(X_comp_original).float()
        y_train = torch.tensor(y)

        train_dataset = Data.TensorDataset(X_train_bank, X_train_sino, X_train_comp, y_train)

        # create dataloader
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,)
        
        if validation_data is not None:
            assert len(validation_data) == 2
            (X_val_bank, X_val_sino, X_val_comp), y_val = validation_data

        model = self.train()
        optimizer = RAdam(model.parameters(), weight_decay=0., lr=2e-3)

        if self.prob_loss:
            loss_func = cross_entropy_with_probs
        else:
            loss_func = F.cross_entropy

        best_acc_val = 0
        for epoch in range(epochs):
            for step, (X_bank, X_sino, X_comp, sino_y) in enumerate(train_loader):
                model.train()
                X_bank, X_sino, X_comp, sino_y = X_bank.to(self.device), X_sino.to(
                    self.device), X_comp.to(self.device), sino_y.to(self.device)

                pred_values = model(X_bank, X_sino, X_comp)
                y_pred = pred_values[0]

                total_loss = 0
                for i in range(6):
                    if self.prob_loss:
                        total_loss += loss_func(y_pred[:, i], sino_y[:, i].float(), reduction="mean")
                    else:
                        total_loss += loss_func(y_pred[:, i], sino_y[:, i].long(), reduction="mean")

                optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                optimizer.step()

            
            if validation_data is None:
                model = self.eval()
                acc_train = model.evaluate((X_bank_original, X_sino_original, X_comp_original), y)
                
                print("Epochs:{} | Training ACC: {:5f}".format(epoch, acc_train))
                if epoch == epochs-1:
                    torch.save(model, model_path + 'MATE_model.pt')
            else:
                model = self.eval()
                acc_train = model.evaluate((X_bank_original, X_sino_original, X_comp_original), y)
                acc_val = model.evaluate((X_val_bank, X_val_sino, X_val_comp), y_val)

                if not save_best:
                    if epoch == epochs - 1:
                        torch.save(model, model_path + 'MATE_model.pt')
                else:
                    if acc_val >= best_acc_val:
                        best_acc_val = acc_val
                        torch.save(model, model_path + 'MATE_model.pt')
                print("Epochs:{} | Training ACC: {:5f} | Validation ACC:{:5f}".format(epoch, acc_train, acc_val))
        print('End of MATE training.')