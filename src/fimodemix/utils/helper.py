from importlib import import_module

def create_class_instance(class_full_path: str, kwargs, *args):
    """Create an instance of a given class.

    :param module_name: where the class is located
    :param kwargs: arguments needed for the class constructor
    :returns: instance of 'class_name'

    """
    module_name, class_name = class_full_path.rsplit(".", 1)
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    if kwargs is None:
        instance = clazz(*args)
    else:
        instance = clazz(*args, **kwargs)

    return instance