from torch import nn


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super().__init__()
        conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation)
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += self.build_conv_block(dim, padding_type, norm_layer, activation)
        self.conv_block = nn.Sequential(*conv_block)

    def build_conv_block(self, dim, padding_type, norm_layer, activation):
        conv_block = []

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p)]
        if norm_layer is not None:
            conv_block += [norm_layer(dim)]
        conv_block += [activation]
        return conv_block

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



