import mindspore.ops as ops
import mindspore.nn as nn
from .resnet import resnet50,resnet101
from .conv_bn_relu import conv_bn_relu
import mindspore.nn as nn

class FpnTopDown(nn.Cell):
    """
    Fpn to extract features
    """
    def __init__(self, in_channel_list, out_channels):
        super(FpnTopDown, self).__init__()
        self.lateral_convs_list_ = []
        self.fpn_convs_ = []
        for channel in in_channel_list:
            l_conv = nn.Conv2d(channel, out_channels, kernel_size=1, stride=1,
                               has_bias=True, padding=0, pad_mode='same')
            fpn_conv = conv_bn_relu(out_channels, out_channels, kernel_size=3, stride=1, depthwise=False)
            self.lateral_convs_list_.append(l_conv)
            self.fpn_convs_.append(fpn_conv)
        self.lateral_convs_list = nn.layer.CellList(self.lateral_convs_list_)
        self.fpn_convs_list = nn.layer.CellList(self.fpn_convs_)
        self.num_layers = len(in_channel_list)

    def construct(self, inputs):
        image_features = ()
        for i, feature in enumerate(inputs):
            image_features = image_features + (self.lateral_convs_list[i](feature),)

        features = (image_features[-1],)
        for i in range(len(inputs) - 1):
            top = len(inputs) - i - 1
            down = top - 1
            size = ops.shape(inputs[down])
            top_down = ops.ResizeBilinear((size[2], size[3]))(features[-1])
            top_down = top_down + image_features[down]
            features = features + (top_down,)

        extract_features = ()
        num_features = len(features)
        for i in range(num_features):
            extract_features = extract_features + (self.fpn_convs_list[i](features[num_features - i - 1]),)

        return extract_features

class BottomUp(nn.Cell):
    """
    Bottom Up feature extractor
    """
    def __init__(self, levels, channels, kernel_size, stride):
        super(BottomUp, self).__init__()
        self.levels = levels
        bottom_up_cells = [
            conv_bn_relu(channels, channels, kernel_size, stride, False) for x in range(self.levels)
        ]
        self.blocks = nn.CellList(bottom_up_cells)

    def construct(self, features):
        for block in self.blocks:
            features = features + (block(features[-1]),)
        return features

class FeatureSelector(nn.Cell):
    """
    Select specific layers from an entire feature list
    """
    def __init__(self, feature_idxes):
        super(FeatureSelector, self).__init__()
        self.feature_idxes = feature_idxes

    def construct(self, feature_list):
        selected = ()
        for i in self.feature_idxes:
            selected = selected + (feature_list[i],)
        return selected

class ResNetV1Fpn(nn.Cell):
    """
    ResNet with FPN as PAA backbone.
    """
    def __init__(self, resnet):
        super(ResNetV1Fpn, self).__init__()
        self.resnet = resnet
        self.fpn = FpnTopDown([512, 1024, 2048], 256)
        self.bottom_up = BottomUp(2, 256, 3, 2)

    def construct(self, x):
        # _, _, c3, c4, c5 = self.resnet(x)
        c3, c4, c5 = self.resnet(x)
        features = self.fpn((c3, c4, c5))
        features = self.bottom_up(features)
        return features


def resnet50_fpn():
    resnet = resnet50()
    return ResNetV1Fpn(resnet)

def resnet101_fpn():
    resnet = resnet101()
    return ResNetV1Fpn(resnet)
