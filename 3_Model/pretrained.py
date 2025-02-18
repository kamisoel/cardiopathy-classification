import torch
from numpy.random.mtrand import Sequence
from torch import nn
import torch.nn.functional as F
import re

from monai.networks.nets import resnet
from monai.utils import ensure_tuple

class PretrainedResNet(nn.Module):
    """
    Implements a ResNet model with pretrained weights using Monai's ResNetFeatures/ResNetEncoder class
    Parameters:
        model_name (string): name of the pretrained ResNet model (e.g. resnet18)
        num_classes (int): number of output classes
        in_channels (int): number of input channels
        hidden_sizes (list of int): number of neurons in fully connected layers after the pretrained model
        use_hierachical_features (bool): whether to use hierarchical features or only the last layer's
        freeze_backbone (bool): whether to freeze backbone layers or not
    """
    def __init__(self, model_name: str, num_classes: int, in_channels: int = 1,
                 hidden_sizes: int | Sequence[int] = (128), use_hierachical_features=False,
                 freeze_backbone: bool = True):
        super().__init__()
        search_res = re.search(r"resnet(\d+)", model_name)
        if search_res:
            self.resnet_depth = int(search_res.group(1))
        else:
            raise ValueError("Illegal model_name, should be e.g. resnet18")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_hierachical_features = use_hierachical_features
        self.backbone = resnet.ResNetEncoder(model_name)
        # self.backbone = monai.networks.nets.resnet18(pretrained=True, shortcut_type="A",
        #                      feed_forward=False, bias_downsample=True, n_input_channels=1)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # if self.resnet_depth in [10, 18, 34]:
        #     expansion = 1
        # else:
        #     expansion = 4
        #in_planes = resnet.get_inplanes()
        # fc_in = in_planes[-1] * expansion * self.in_channels
        in_planes = self.backbone.num_channels_per_output()[self.backbone.backbone_names.index(model_name)]
        if use_hierachical_features:
            fc_in = torch.sum(torch.tensor(in_planes)) * self.in_channels
        else:
            fc_in = in_planes[-1] * self.in_channels
        hidden_sizes = [fc_in, *ensure_tuple(hidden_sizes), num_classes]

        self.fc_layers = nn.ModuleList([torch.nn.Linear(cin, cout) for cin, cout in zip(hidden_sizes, hidden_sizes[1:])])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False


    def forward(self, x):
        n_batches = x.shape[0]
        # backbone takes only 1 in_channel so we map them on the batch dimension
        x = x.view((-1, 1, *x.shape[2:]))
        xs = self.backbone(x)  # ResNetFeatures return hierachical features as a list
        #print([self.avg_pool(f).shape for f in x])
        if self.use_hierachical_features:
            x = torch.concat([self.avg_pool(x) for x in xs], dim=1)
        else:
            x = self.avg_pool(xs[-1])             # for classification the last layers result is enough
        x = x.view([n_batches, -1])
        for fc in self.fc_layers:
            x = fc(F.relu(x))
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)