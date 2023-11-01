import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding_mode='reflect', act_fun=nn.ELU, normalization=nn.InstanceNorm2d):
        super().__init__()
        self.pad_mode = padding_mode
        self.filter_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        n_pad_pxl = int(self.dilation * (self.filter_size - 1) / 2)

        # this is for backward campatibility with older model checkpoints
        self.block = nn.ModuleDict(
            {
                'conv_f': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_f': act_fun(),
                'conv_m': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_m': nn.Sigmoid(),
                'norm': normalization(out_channels)
            }
        )

        self.ac = nn.ReLU()

    def forward(self, x, *args, **kwargs):
        features = self.block.act_f(self.block.conv_f(x))
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)
        output = self.ac(output)

        return output


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=GatedBlock):
        super().__init__()

        self.conv = conv_block(in_channels, out_channels)
        self.down = nn.AvgPool2d(2, 2)

    def forward(self, inputs, mask=None):
        outputs = self.down(inputs)
        outputs = self.conv(outputs, mask=mask)
        return outputs


class UpsampleBlock(nn.Module):
    def __init__(self, out_channels, upsample_mode, num_filt, conv_block=GatedBlock):
        super().__init__()

        #  = out_channels if same_num_filt else out_channels * 2
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(
                num_filt, out_channels, 4, stride=2, padding=1)
            self.conv = conv_block(out_channels * 2, out_channels)
        elif upsample_mode == 'bilinear' or upsample_mode == 'nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    nn.Conv2d(num_filt, out_channels,
                                              3, padding=1)
                                    )
            self.conv = conv_block(out_channels * 2, out_channels)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)
        output = self.conv(torch.cat([in1_up, inputs2], 1))

        return output

# Mask prediction network
class MPN(nn.Module):
    def __init__(self, in_dim=3, upsample_mode='nearest', udim='pp', U=4):
        super().__init__()
        self.U = U
        out_dim = in_dim

        if udim == 'pp':
            filters = [16, 32, 48, 64, 80]
        elif udim == 'npbg':
            filters = [64, 128, 256, 512, 1024]
            filters = [x // 4 for x in filters] # [16, 32, 64, 128, 256]
        elif udim == '4xnpbg':
            filters = [64, 128, 256, 512, 1024]
        else:
            assert False

        self.start = GatedBlock(in_dim, filters[0])

        self.down1 = DownsampleBlock(filters[0], filters[1])
        self.down2 = DownsampleBlock(filters[1], filters[2])

        if U == 4:
            self.down3 = DownsampleBlock(filters[2], filters[3])
            self.down4 = DownsampleBlock(filters[3], filters[4])

            self.up4 = UpsampleBlock(filters[3], upsample_mode, filters[4])
            self.up3 = UpsampleBlock(filters[2], upsample_mode, filters[3])
        self.up2 = UpsampleBlock(filters[1], upsample_mode, filters[2])
        self.up1 = UpsampleBlock(filters[0], upsample_mode, filters[1])

        self.final = nn.Sequential(
            nn.Conv2d(filters[0], out_dim, 1),
            # nn.GroupNorm(num_groups=1, num_channels=in_dim)
        )

    def forward(self, x):

        in64 = self.start(x)

        down1 = self.down1(in64)
        down2 = self.down2(down1)
        if self.U == 4:
            down3 = self.down3(down2)
            down4 = self.down4(down3)

            up4 = self.up4(down4, down3)
            up3 = self.up3(up4, down2)
            up2 = self.up2(up3, down1)
        else:
            up2 = self.up2(down2, down1)
        up1 = self.up1(up2, in64)

        final = self.final(up1)  # [1, 3, 224, 224]
        # final_n = final / final.sum(dim=1, keepdim=True) # perform very well, but prone to triger nan
        # final_n = torch.sigmoid(final) # perform bad
        final_n = torch.softmax(final, dim=1)
        

        # return final_n.mul(x)
        return final_n


if __name__ == '__main__':
    device = torch.device("cuda")
    moda_num = 64 
    batch_size = 1
    H = 800
    W = 800
    input_size = (batch_size, moda_num, H, W)

    data = torch.randn(input_size).to(device)
    unet = MPN(U=2, udim='pp', in_dim=moda_num).to(device)
    ret = unet(data)
    print(f"input size: {data.size()}") # torch.Size([1, 3, 224, 224])
    print(f"output size: {ret.size()}") # torch.Size([1, 3, 224, 224])

    print(summary(unet, input_size[1:]))