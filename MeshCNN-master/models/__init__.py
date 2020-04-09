def create_model(opt):
    from .mesh_classifier import ClassifierModel
    model = ClassifierModel(opt)
    return model
