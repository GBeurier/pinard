import os
import importlib
from .utils import TF_AVAILABLE, TORCH_AVAILABLE
import inspect


# 1. str
# "/myexp/cnn.pt"
# "sklearn.linear_model.ElasticNet"

# 2. instance
# RandomForestClassifier(n_estimators=50, max_depth=5)

# 3. dict
# {'class': 'sklearn.linear_model.ElasticNet', 'params': {'alpha': 0.1}}
# {'import': 'custom_lib.model.CustomModel', 'params': {...}}
# {'function': get_my_cnn_tf_model, 'params': {...}}

# 4. callable
# get_my_cnn_tf_model
# sklearn.linear_model.ElasticNet

class ModelBuilderFactory:
    @staticmethod
    def build_models(model_config, dataset, task, force_params={}):
        """
        Build and return a list of model instances based on the provided model_config.

        Parameters:
        - model_config: Can be one of the following:

            1. A string representing a file path to a saved model (e.g., '/myexp/cnn.pt', 'PLS.pkl').
            2. An instance of a model (e.g., an already instantiated model object).
            3. A dict with 'class' and 'params' keys, where 'class' is a string import path
               (e.g., {'class': 'sklearn.linear_model.ElasticNet', 'params': {'alpha': 0.1}}).
               (e.g., {'import': 'custom_lib.model.CustomModel', 'params': {...}}).
               (e.g., {'function': get_my_cnn_tf_model, 'params': {...}}).
            4. A callable function (that returns an instance of a model (e.g., get_my_cnn_tf_model), or class).

        - dataset: The dataset object to be used for model preparation or training.

        Returns:
        - A list of model instances.
        """
        nb_folds = len(dataset.folds) if dataset.folds is not None else 1
        models = []

        # Build the initial model
        model = ModelBuilderFactory.build_single_model(model_config, dataset, task, force_params=force_params)
        framework = ModelBuilderFactory.detect_framework(model)
        models.append(model)
        
        if nb_folds > 1:
            for _ in range(nb_folds - 1):
                model_copy = ModelBuilderFactory._clone_model(model, framework)
                models.append(model_copy)

        return models, framework

    @staticmethod
    def build_single_model(model_config, dataset, task, force_params={}):
        if task == "classification":
            force_params['num_classes'] = dataset.num_classes  # TODO get loss to applied num_classes (sparse_categorical_crossentropy = 1, categorical_crossentropy = num_classe)
            # force_params['num_classes'] = 1

        if isinstance(model_config, str):  # 1
            # print("Building from string")
            return ModelBuilderFactory._from_string(model_config, force_params)
        
        elif isinstance(model_config, dict):  # 3
            # print("Building from dict")
            return ModelBuilderFactory._from_dict(model_config, dataset, force_params)

        elif hasattr(model_config, '__class__') and not inspect.isclass(model_config) and not inspect.isfunction(model_config):  # 2
            # print("Building from instance")
            return ModelBuilderFactory._from_instance(model_config)

        elif callable(model_config):  # 4
            # print("Building from callable")
            return ModelBuilderFactory._from_callable(model_config, dataset, force_params)

        else:
            raise ValueError("Invalid model_config format.")

    @staticmethod
    def _from_string(model_str, force_params=None):
        if os.path.exists(model_str):
            model = ModelBuilderFactory._load_model_from_file(model_str)
            if force_params is not None:
                model = ModelBuilderFactory.reconstruct_object(model, force_params)
            return model
        
        else:
            try:
                cls = ModelBuilderFactory.import_class(model_str)
                model = ModelBuilderFactory.prepare_and_call(cls, force_params)
            except Exception as e:
                raise ValueError(f"Invalid model string format: {str(e)}") from e

    @staticmethod
    def _from_instance(model_instance, force_params=None):
        if force_params is not None:
            model_instance = ModelBuilderFactory.reconstruct_object(model_instance, force_params)
        return model_instance

    @staticmethod
    def _from_dict(model_dict, dataset, force_params=None):
        if 'class' in model_dict:
            class_path = model_dict['class']
            params = model_dict.get('params', {})
            cls = ModelBuilderFactory.import_class(class_path)
            model = ModelBuilderFactory.prepare_and_call(cls, params, force_params)
            return model

        elif 'import' in model_dict:
            object_path = model_dict['import']
            params = model_dict.get('params', {})
            obj = ModelBuilderFactory.import_object(object_path)
            
            if callable(obj):  # function or class
                model = ModelBuilderFactory.prepare_and_call(obj, params, force_params)
            else:  # instance
                model = obj
                if force_params is not None:
                    model = ModelBuilderFactory.reconstruct_object(model, params, force_params)
            
            return model

        elif 'function' in model_dict:
            callable_model = model_dict['function']
            params = model_dict.get('params', {})
            framework = model_dict.get('framework', None)
            if framework is None:
                framework = getattr(callable_model, 'framework', None)
            
            if framework is None:
                raise ValueError("Cannot determine framework from callable model_config. Please set 'experiments.utils.framework' decorator on the function or add 'framework' key to the config.")
            
            input_dim = ModelBuilderFactory._get_input_dim(framework, dataset)
            params['input_dim'] = input_dim
            params['input_shape'] = input_dim
            
            model = ModelBuilderFactory.prepare_and_call(callable_model, params, force_params)
            return model

        else:
            raise ValueError("Dict model_config must contain 'class', 'path', or 'callable' with 'framework' key.")

    @staticmethod
    def _from_callable(model_callable, dataset, force_params=None):
        framework = None
        if inspect.isclass(model_callable):
            framework = ModelBuilderFactory.detect_framework(model_callable)
        elif inspect.isfunction(model_callable):
            framework = getattr(model_callable, 'framework', None)
        
        if framework is None:
            raise ValueError("Cannot determine framework from callable model_config. Please set 'experiments.utils.framework' decorator on the callable.")
        input_dim = ModelBuilderFactory._get_input_dim(framework, dataset)
        params = {"input_dim": input_dim, "input_shape": input_dim}
        
        model = ModelBuilderFactory.prepare_and_call(model_callable, params, force_params)
        return model

    @staticmethod
    def _clone_model(model, framework):
        """
        Clone the model using framework-specific cloning methods.

        Returns:
        - A cloned model instance.
        """
        if framework == 'sklearn':
            from sklearn.base import clone
            return clone(model)

        elif framework == 'tensorflow':
            if TF_AVAILABLE:
                from tensorflow.keras.models import clone_model
                cloned_model = clone_model(model)
                print("Model cloned")
                return cloned_model

        # elif framework == 'pytorch':
        #     import torch
        #     from copy import deepcopy
        #     # Deepcopy works for PyTorch models in most cases
        #     return deepcopy(model)

        else:
            # Fallback to deepcopy
            from copy import deepcopy
            return deepcopy(model)

    @staticmethod
    def _get_input_dim(framework, dataset):
        if framework == 'tensorflow':
            input_dim = dataset.x_train_('union').shape[1:]
        elif framework == 'sklearn':
            # input_dim = dataset.x_train.shape[1:]
        # elif framework == 'pytorch':
            input_dim = dataset.x_train.shape[1:]
        else:
            raise ValueError("Unknown framework.")
        return input_dim

    @staticmethod
    def import_class(class_path):
        module_name, class_name = class_path.rsplit('.', 1)
        if module_name.startswith('tensorflow'):
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is not available but required to load this model.")
        elif module_name.startswith('torch'):
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available but required to load this model.")
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls

    @staticmethod
    def import_object(object_path):
        module_name, object_name = object_path.rsplit('.', 1)
        if module_name.startswith('tensorflow'):
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is not available but required to load this model.")
        elif module_name.startswith('torch'):
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available but required to load this model.")
        module = importlib.import_module(module_name)
        obj = getattr(module, object_name)
        return obj

    @staticmethod
    def detect_framework(model):
        """
        Detect the framework from the model instance.

        Returns:
        - A string representing the framework.
        """
        
        if inspect.isclass(model):
            model_desc = f"{model.__module__}.{model.__name__}"
        else:
            model_desc = f"{model.__class__.__module__}.{model.__class__.__name__}"
        if TF_AVAILABLE and 'tensorflow' in model_desc:
            return 'tensorflow'
        if TF_AVAILABLE and 'keras' in model_desc:
            return 'tensorflow'
        elif TORCH_AVAILABLE and 'torch' in model_desc:
            return 'pytorch'
        elif 'sklearn' in model_desc:
            return 'sklearn'
        else:
            raise ValueError("Cannot determine framework from the model instance.")

    @staticmethod
    def _filter_params(model_or_class, params):
        constructor_signature = inspect.signature(model_or_class.__init__)
        valid_params = {param.name for param in constructor_signature.parameters.values() if param.name != 'self'}
        filtered_params = {key: value for key, value in params.items() if key in valid_params}
        return filtered_params

    @staticmethod
    def _force_param_on_instance(model, force_params):
        try:
            filtered_params = ModelBuilderFactory._filter_params(model, force_params)
            new_model = model.__class__(**filtered_params)
            return new_model
        except Exception as e:
            print(f"Warning: Cannot force parameters on the model instance. Reason: {e}")
            return model

    @staticmethod
    def prepare_and_call(callable_obj, params=None, force_params=None):
        """
        Prepare parameters for the callable and invoke it. Parameters are chosen from `force_params` first, 
        then from `params`. If a required parameter is missing, an exception is raised.

        Parameters:
        - callable_obj: The callable (function or class) to be invoked.
        - params: A dictionary of default parameters (can be None).
        - force_params: A dictionary of parameters that override `params` (can be None).

        Returns:
        - The result of calling `callable_obj` with the prepared parameters.
        
        Raises:
        - TypeError: If a required parameter is missing.
        """
        if params is None:
            params = {}
        if force_params is None:
            force_params = {}
            
        merged_params = {**params, **force_params}

        # Get the signature of the callable
        signature = inspect.signature(callable_obj)

        # Dictionary to hold the final arguments
        final_args = {}

        # Iterate over the parameters of the callable
        for name, param in signature.parameters.items():
            if name == 'self':  # Skip 'self' for instance methods or constructors
                continue

            if name in force_params:
                final_args[name] = force_params[name]
            elif name in params:
                final_args[name] = params[name]
            elif param.default is not inspect.Parameter.empty:
                final_args[name] = param.default
            elif name == "params" or name == "force_params":
                final_args[name] = merged_params
            else:
                # If the parameter is required and not provided, raise an exception
                raise TypeError(f"Missing required parameter: '{name}'")

        # Call the callable with the prepared arguments
        return callable_obj(**final_args)

    @staticmethod
    def reconstruct_object(obj, params=None, force_params=None):
        """
        Reconstruct an object using its current attributes as default values,
        then overwriting with provided params and force_params.

        Parameters:
        - obj: The object to be reconstructed.
        - params: A dictionary of parameters to overwrite the object's current parameters.
        - force_params: A dictionary of parameters that take precedence over both the object's
                        current parameters and params.

        Returns:
        - A new instance of the object with the updated parameters.
        
        Raises:
        - TypeError: If a required parameter is missing and no default value is provided.
        """
        if params is None:
            params = {}
        if force_params is None:
            force_params = {}
            
        merged_params = {**params, **force_params}

        cls = obj.__class__
        signature = inspect.signature(cls)
        current_params = obj.__dict__.copy()  # This assumes the object stores its state in __dict__

        final_args = {}
        
        for name, param in signature.parameters.items():
            if name == 'self':  # Skip 'self'
                continue

            if name in force_params:
                final_args[name] = force_params[name]
            elif name in params:
                final_args[name] = params[name]
            elif name in current_params:
                final_args[name] = current_params[name]
            elif param.default is not inspect.Parameter.empty:
                final_args[name] = param.default
            elif name == "params" or name == "force_params":
                final_args[name] = merged_params
            else:
                raise TypeError(f"Missing required parameter: '{name}'")

        return cls(**final_args)

    @staticmethod
    def _load_model_from_file(model_path):
        """
        Load a model from a file path.

        Returns:
        - A tuple of (model instance, framework string).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' does not exist.")
        _, ext = os.path.splitext(model_path)

        # TensorFlow model
        if ext in ['.h5', '.hdf5', '.keras']:
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is not available but required to load this model.")
            from tensorflow import keras
            # from tensorflow.keras import metrics

            # Pass custom objects if needed
            custom_objects = {
                # 'mse': metrics.MeanSquaredError()
            }

            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            return model

        # PyTorch model
        elif ext == '.pt':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available but required to load this model.")
            import torch
            model = torch.load(model_path)
            return model

        # Sklearn model
        elif ext == '.pkl':
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model

        else:
            raise ValueError(f"Unsupported file extension '{ext}' for model file.")
