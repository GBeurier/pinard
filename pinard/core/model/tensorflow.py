class TFModelBuilder:
    def __init__(self, model_params):
        self.model_params = model_params

    def build_model(self):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout

        model = Sequential()
        for layer in self.model_params['layers']:
            if layer['type'] == 'Dense':
                model.add(Dense(units=layer['units'], activation=layer['activation']))
            elif layer['type'] == 'Dropout':
                model.add(Dropout(rate=layer['rate']))

        model.compile(optimizer=self.model_params['optimizer'], loss=self.model_params['loss'])
        return model

class TFModelManager:
    def __init__(self, models=None, model_config=None):
        self.models = models
        self.model_config = model_config
        self.framework = 'tensorflow'

    def train(self, dataset, training_params, input_dim=None, metrics=None, no_folds=False):
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

        models = [self.models[0]] if no_folds else self.models
        for (x_train, y_train, x_val, y_val), model in zip(dataset.fold_data('union', no_folds), models):
            print(f"Training fold with shapes:", x_train.shape, y_train.shape, x_val.shape, y_val.shape)
            loss = training_params.get('loss', 'mse')

            x_train = tf.convert_to_tensor(x_train)
            y_train = tf.convert_to_tensor(y_train)
            x_val = tf.convert_to_tensor(x_val)
            y_val = tf.convert_to_tensor(y_val)

            model.compile(optimizer=training_params.get('optimizer', 'adam'),
                          loss=loss,
                          metrics=metrics)

            callbacks = []
            if training_params.get('early_stopping', True):
                callbacks.append(EarlyStopping(monitor='val_loss', patience=training_params.get('patience', 50)))

            if training_params.get('cyclic_lr', False):
                base_lr = training_params.get('base_lr', 1e-4)
                max_lr = training_params.get('max_lr', 1e-2)
                step_size = training_params.get('step_size', 2000)

                def cyclic_lr(epoch):
                    cycle = np.floor(1 + epoch / (2 * step_size))
                    x = np.abs(epoch / step_size - 2 * cycle + 1)
                    lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
                    return lr

                callbacks.append(LearningRateScheduler(cyclic_lr))

            model.fit(x_train, y_train,
                      validation_data=(x_val, y_val),
                      epochs=training_params.get('epochs', 100),
                      batch_size=training_params.get('batch_size', 32),
                      callbacks=callbacks,
                      verbose=training_params.get('verbose', 1))

    def predict(self, dataset, task, return_all=False, no_folds=False, raw_class_output=False):
        y_pred = None
        if len(self.models) == 1 or no_folds:
            y_pred = self.models[0].predict(dataset.x_test_('union'))
            if task == 'classification' and not raw_class_output:
                return np.argmax(y_pred, axis=1)
            else:
                return y_pred
        else:
            y_preds = [model.predict(dataset.x_test_('union')) for model in self.models]
            if task == 'classification':
                if return_all:
                    if raw_class_output:
                        return y_preds  # Return raw probabilities/logits for each fold
                    y_preds = [np.argmax(y_pred, axis=1) for y_pred in y_preds]
                    return y_preds
                else:
                    y_preds = np.mean(y_preds, axis=0)
                    return np.argmax(y_preds, axis=1) if not raw_class_output else y_preds
            else:
                return y_preds if return_all else np.mean(y_preds, axis=0)

    def save_model(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for i, model in enumerate(self.models):
            model.save(path + f'_{i}.keras')

    def load_model(self, path):
        import os
        from tensorflow.keras.models import load_model
        if os.path.exists(path):
            self.models = []
            for i in range(len(os.listdir(path))):
                if os.path.exists(path + f'model_{i}.keras'):
                    self.models.append(load_model(path + f'model_{i}.keras'))
        else:
            raise FileNotFoundError(f"Model file not found at: {path}")