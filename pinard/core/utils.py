# utils.py
# Global availability flags for frameworks
import importlib

TF_AVAILABLE = importlib.util.find_spec('tensorflow') is not None
TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None

def framework(framework_name):
    def decorator(func):
        func.framework = framework_name
        return func
    return decorator


def get_full_import_path(instance):
    return f"{instance.__class__.__module__}.{instance.__class__.__name__}"


def get_construction_params(instance):
    if hasattr(instance, 'get_params'):
        return instance.get_params()
    else:
        # If there's no `get_params()` method, fallback to using __dict__
        return instance.__dict__

def deserialize_object(serialized_obj):
    """Reconstruct the object from its serialized form."""
    module_name, class_name = serialized_obj['class'].rsplit('.', 1)
    module = importlib.import_module(module_name)
    obj_class = getattr(module, class_name)
    if 'params' in serialized_obj:
        return obj_class(**serialized_obj['params'])
    return obj_class()

def deserialize_pipeline(serialized_pipeline):
    """Recursively deserialize the pipeline."""
    if isinstance(serialized_pipeline, list):
        return [deserialize_pipeline(step) for step in serialized_pipeline]
    if isinstance(serialized_pipeline, dict):
        return {key: deserialize_pipeline(value) for key, value in serialized_pipeline.items()}
    return deserialize_object(serialized_pipeline)
