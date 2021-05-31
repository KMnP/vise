# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Optional list of dependencies required by the package
dependencies = ['torch', 'torchvision']

import torch
import torchvision


model_urls = {
    "resnet50_1m": "https://cornell.box.com/shared/static/r36nd2o0w5ch6ujuaxj0mtasxaqg0l5t.pth",
    "resnet50_250m": "https://cornell.box.com/shared/static/y210vs3iktungg7wrf72ibzl87jrojna.pth",
}


def _resnet50(model_arch: str, pretrained: bool = False, **kwargs):
    """
    Args:
        model_arch (str): specify which model file to download.
        progress (bool): If True, displays a progress bar of the download to stderr.
    """

    # Create a torchvision resnet50 with randomly initialized weights.
    model = torchvision.models.resnet50(pretrained=False, **kwargs)

    # Get the model before the global aver-pooling layer.
    model = torch.nn.Sequential(*list(model.children())[:-2])

    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            model_urls[model_arch], progress=True)
        )
    return model


def resnet50_1m(pretrained: bool = False, **kwargs):
    """
    Constructs a ResNet-50 model pre-trained on 1.2M visual engagement
    data in `"Exploring Visual Engagement Signals for Representation Learning"
    <https://arxiv.org/abs/2104.07767>`_

    This is a torchvision-like model. Given a batch of image tensors with size
    ``(B, 3, 224, 224)``, this model computes spatial image features of size
    ``(B, 2048, 7, 7)``, where B = batch size.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    model = _resnet50(
        model_arch="resnet50_1m", pretrained=pretrained, **kwargs
    )
    return model


def resnet50_250m(pretrained: bool = False, **kwargs):
    """
    Constructs a ResNet-50 model pre-trained on 250M visual engagement
    data in `"Exploring Visual Engagement Signals for Representation Learning"
    <https://arxiv.org/abs/2104.07767>`_

    This is a torchvision-like model. Given a batch of image tensors with size
    ``(B, 3, 224, 224)``, this model computes spatial image features of size
    ``(B, 2048, 7, 7)``, where B = batch size.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    model = _resnet50(
        model_arch="resnet50_250m", pretrained=pretrained, **kwargs
    )
    return model

