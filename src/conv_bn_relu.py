import mindspore.nn as nn

def conv_bn_relu(in_channel, out_channel, kernel_size, stride, depthwise, activation='relu6'):
    output = []
    output.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode="same",
                            group=1 if not depthwise else in_channel))
    output.append(nn.BatchNorm2d(out_channel))
    if activation:
        output.append(nn.get_activation(activation))
    return nn.SequentialCell(output)