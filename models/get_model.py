from torchvision.models import resnet18, resnet34

from .resnet_small import resnet20, resnet32, resnet44, resnet56


def get_model(model_name, **model_kwargs):
    if model_name == "resnet20":
        return resnet20(**model_kwargs)
    elif model_name == "resnet32":
        return resnet32(**model_kwargs)
    elif model_name == "resnet44":
        return resnet44(**model_kwargs)
    elif model_name == "resnet56":
        return resnet56(**model_kwargs)
    elif model_name == "resnet18":
        return resnet18(**model_kwargs)
    elif model_name == "resnet34":
        return resnet34(**model_kwargs)
    else:
        raise NotImplementedError
