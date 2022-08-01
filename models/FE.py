import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from ..optimizers.radam import RAdam
from sklearn.metrics import accuracy_score, mean_squared_error


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


# Finalcial Embedding Model
class FE(nn.Module):  
    def __init__(self, max_leaf, leaf_dim, n_estimators=30, hidden_dim=12, device="cuda:0"):
        super(FE, self).__init__()
        self.max_leaf = max_leaf
        self.leaf_dim = leaf_dim
        self.n_estimators = n_estimators
        self.hidden_dim = hidden_dim
        self.out_dim = 1
        self.device = device
        self.mish = Mish()  # nn.LeakyReLU()

        # embedding layers
        self.leaf_embedding = nn.Embedding(self.max_leaf, self.leaf_dim)

        self.conv_layer_cls = nn.Conv1d(
            in_channels=self.n_estimators,
            out_channels=self.n_estimators * self.hidden_dim,
            kernel_size=self.leaf_dim,
            groups=self.n_estimators,
        )
        self.conv_layer_cls_final = nn.Conv1d(
            in_channels=self.n_estimators,
            out_channels=self.n_estimators * self.out_dim,
            kernel_size=self.hidden_dim,
            groups=self.n_estimators,
        )

        self.conv_layer_reg = nn.Conv1d(
            in_channels=self.n_estimators,
            out_channels=self.n_estimators * self.hidden_dim,
            kernel_size=self.leaf_dim,
            groups=self.n_estimators,
        )
        self.conv_layer_reg_final = nn.Conv1d(
            in_channels=self.n_estimators,
            out_channels=self.n_estimators * self.out_dim,
            kernel_size=self.hidden_dim,
            groups=self.n_estimators,
        )

    def forward(self, X):
        # shape: torch.Size([batch, n_estimator, leaf_dim])
        leaf_embedding = self.leaf_embedding(X)

        # shape: torch.Size([batch, n_estimator * self.hidden_dim, 1])
        cls_out = self.mish(self.conv_layer_cls(leaf_embedding))

        # shape: torch.Size([batch, n_estimator, self.hidden_dim])
        cls_out = cls_out.view(cls_out.shape[0], self.n_estimators, -1)

        # shape: torch.Size([batch, n_estimator, 1])
        cls_out = self.conv_layer_cls_final(cls_out)

        cls_out = torch.mean(cls_out, dim=1)
        cls_out = torch.sigmoid(cls_out)

        reg_out = self.mish(self.conv_layer_reg(leaf_embedding))
        reg_out = reg_out.view(reg_out.shape[0], self.n_estimators, -1)
        reg_out = self.conv_layer_cls_final(reg_out)
        reg_out = torch.mean(reg_out, dim=1)
        reg_out = torch.relu(reg_out)

        return cls_out, reg_out, leaf_embedding

    def evaluate(self, X_leaves, y_cls, y_reg):
        y_pred_cls, y_pred_reg = self.predict(X_leaves)

        acc = accuracy_score(y_cls, y_pred_cls)
        mse = mean_squared_error(y_reg, y_pred_reg)
        return acc, mse

    def predict(self, X_leaves):
        model = self.eval()

        X_leaves = torch.tensor(X_leaves).long()
        test_dataset = Data.TensorDataset(X_leaves)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False,)

        y_pred_cls, y_pred_reg = [], []

        for step, (X_leaves,) in enumerate(test_loader):
            X_leaves = X_leaves.to(self.device)
            cls_out, reg_out, _ = model(X_leaves)

            cls_out = ((cls_out > 0.5).int()).view(-1)
            y_pred_cls.extend(cls_out.detach().cpu().numpy().tolist())
            y_pred_reg.extend(reg_out.view(-1,).detach().cpu().numpy().tolist())

        return y_pred_cls, y_pred_reg

    def fit(self, X, y, validation_data=None, batch_size=128, epochs=15):
        X_train = torch.tensor(X).long()
        y_train_cls, y_train_reg = torch.tensor(y[0]).float(), torch.tensor(y[1]).float()
        
        train_dataset = Data.TensorDataset(X_train, y_train_cls, y_train_reg)
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,)

        if validation_data is not None:
            assert len(validation_data) == 3
            X_val, y_val_cls, y_val_reg = validation_data

        model = self.train()
        optimizer = RAdam(model.parameters(), weight_decay=0.0, lr=2e-3)
        cls_loss_func = nn.BCELoss()
        reg_loss_func = F.mse_loss  # F.l1_loss

        for epoch in range(epochs):
            for step, (batch_leaves, batch_cls, batch_reg) in enumerate(train_loader):
                model.train()
                batch_leaves, batch_cls, batch_reg = (
                    batch_leaves.to(self.device),
                    batch_cls.to(self.device),
                    batch_reg.to(self.device),
                )
                batch_cls, batch_reg = batch_cls.view(-1, 1), batch_reg.view(-1, 1)

                cls_out, reg_out, _ = model(batch_leaves)

                cls_loss = cls_loss_func(cls_out, batch_cls)
                reg_loss = 10 * reg_loss_func(reg_out, batch_reg)
                loss = cls_loss + reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (step + 1) % 50 == 0:
                    print(
                        "CLS loss:%.4f, REG loss:%.4f, Loss:%.4f"
                        % (cls_loss.item(), reg_loss.item(), loss.item())
                    )
            if validation_data is not None:
                model.eval()
                acc, mse = model.evaluate(X_val, y_val_cls, y_val_reg)
                print("Validation Epochs:{} | Acc:{} | mse:{}".format(epoch, acc, mse))
        print('End of FE training.')
