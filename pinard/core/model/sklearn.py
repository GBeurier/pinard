# class SklearnModelBuilder:
#     def __init__(self, model_params):
#         self.model_params = model_params

#     def build_model(self):
#         from sklearn.linear_model import LinearRegression

#         model = LinearRegression(**self.model_params)
#         return model

# class SklearnModelManager:
#     def __init__(self, models=None, model_config=None):
#         self.models = models
#         self.model_config = model_config
#         self.framework = 'sklearn'

#     def train(self, dataset, training_params, input_dim=None, metrics=None, no_folds=False):
#         models = [self.models[0]] if no_folds else self.models
#         for (x_train, y_train, x_val, y_val), model in zip(dataset.fold_data('union', no_folds), models):
#             print(f"Training fold with shapes:", x_train.shape, y_train.shape, x_val.shape, y_val.shape)
#             model.fit(x_train, y_train)

#     def predict(self, dataset, task, return_all=False, no_folds=False, raw_class_output=False):
#         y_pred = None
#         if len(self.models) == 1 or no_folds:
#             y_pred = self.models[0].predict(dataset.x_test_('union'))
#             return y_pred
#         else:
#             y_preds = [model.predict(dataset.x_test_('union')) for model in self.models]
#             return y_preds if return_all else np.mean(y_preds, axis=0)

#     def save_model(self, path):
#         import os
#         import joblib
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         for i, model in enumerate(self.models):
#             joblib.dump(model, path + f'_{i}.pkl')

#     def load_model(self, path):
#         import os
#         import joblib
#         if os.path.exists(path):
#             self.models = []
#             for i in range(len(os.listdir(path))):
#                 if os.path.exists(path + f'model_{i}.pkl'):
#                     self.models.append(joblib.load(path + f'model_{i}.pkl'))
#         else:
#             raise FileNotFoundError(f"Model file not found at: {path}")