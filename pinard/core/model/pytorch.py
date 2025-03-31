class TorchModelBuilder:
    def __init__(self, model_params):
        self.model_params = model_params

    def build_model(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class SimpleNet(nn.Module):
            def __init__(self, input_size):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)
                self.dropout1 = nn.Dropout(0.2)
                self.dropout2 = nn.Dropout(0.1)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout1(x)
                x = F.relu(self.fc2(x))
                x = self.dropout2(x)
                x = self.fc3(x)
                return x

        input_size = self.model_params.get('input_size', 100)  # Default input size
        model = SimpleNet(input_size)
        return model

class TorchModelManager:
    def __init__(self, models=None, model_config=None):
        self.models = models
        self.model_config = model_config
        self.framework = 'pytorch'

    def train(self, dataset, training_params, input_dim=None, metrics=None, no_folds=False):
        import torch
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        models = [self.models[0]] if no_folds else self.models
        for (x_train, y_train, x_val, y_val), model in zip(dataset.fold_data('union', no_folds), models):
            print(f"Training fold with shapes:", x_train.shape, y_train.shape, x_val.shape, y_val.shape)
            loss_fn = training_params.get('loss', torch.nn.MSELoss())

            x_train = torch.tensor(x_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            x_val = torch.tensor(x_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)

            optimizer = optim.Adam(model.parameters(), lr=training_params.get('lr', 1e-3))

            train_dataset = TensorDataset(x_train, y_train)
            val_dataset = TensorDataset(x_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=training_params.get('batch_size', 32), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=training_params.get('batch_size', 32), shuffle=False)

            best_val_loss = float('inf')
            best_weights = None

            for epoch in range(training_params.get('epochs', 100)):
                model.train()
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(x_batch)
                    loss = loss_fn(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        outputs = model(x_batch)
                        loss = loss_fn(outputs, y_batch)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = model.state_dict()

                print(f"Epoch [{epoch+1}/{training_params.get('epochs', 100)}], Validation Loss: {val_loss:.4f}")

            if best_weights is not None:
                model.load_state_dict(best_weights)

    def predict(self, dataset, task, return_all=False, no_folds=False, raw_class_output=False):
        import torch
        y_pred = None
        if len(self.models) == 1 or no_folds:
            y_pred = self.models[0](torch.tensor(dataset.x_test_('union'), dtype=torch.float32)).detach().numpy()
            if task == 'classification' and not raw_class_output:
                return (y_pred >= 0.5).astype(int)
            else:
                return y_pred
        else:
            y_preds = [model(torch.tensor(dataset.x_test_('union'), dtype=torch.float32)).detach().numpy() for model in self.models]
            if task == 'classification':
                if return_all:
                    if raw_class_output:
                        return y_preds
                    y_preds = [(y_pred >= 0.5).astype(int) for y_pred in y_preds]
                    return y_preds
                else:
                    y_preds = np.mean(y_preds, axis=0)
                    return (y_preds >= 0.5).astype(int) if not raw_class_output else y_preds
            else:
                return y_preds if return_all else np.mean(y_preds, axis=0)

    def save_model(self, path):
        import os
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), path + f'_{i}.pt')

    def load_model(self, path):
        import os
        import torch
        if os.path.exists(path):
            self.models = []
            for i in range(len(os.listdir(path))):
                if os.path.exists(path + f'model_{i}.pt'):
                    model = self.models[0].__class__()
                    model.load_state_dict(torch.load(path + f'model_{i}.pt'))
                    self.models.append(model)
        else:
            raise FileNotFoundError(f"Model file not found at: {path}")