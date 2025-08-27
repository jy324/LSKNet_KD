import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchinfo import summary


def resnet(name, cls_num, pretrained=False):
    if name == 'ResNet18':
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == 'ResNet34':
        model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == 'ResNet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == 'ResNet101':
        model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == 'ResNet152':
        model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported model name: {name}. "
                         f"Please choose one model in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'].")
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, cls_num)
    return model


if __name__ == '__main__':
    model = resnet('ResNet50', cls_num=25, pretrained=False).cuda()
    summary(model, (1, 3, 224, 224))
