import torch
import torch.nn as nn


#complete model architecture overhaul
class U_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder (contracting path)
        self.conv1 = self.contract_block(in_channels, 32)
        self.conv2 = self.contract_block(32, 64)
        self.conv3 = self.contract_block(64, 128)
        self.conv4 = self.dilated_block(128, 256, dilation=2)
        self.conv5 = self.dilated_block(256, 512, dilation=4)

        # Bottleneck (stays at conv3/conv4/conv5's resolution, H/8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # Decoder (expanding path)
        # upconv6/5/4: fuse skip connections that are still at H/8 - no
        # upsampling here, conv5/conv4/conv3 were never downsampled further.
        self.upconv6 = self.fuse_block(1024 + 512, 512)
        self.upconv5 = self.fuse_block(512 + 256, 256)
        self.upconv4 = self.fuse_block(256 + 128, 128)

        # upconv3/2/1: real spatial upsampling, back through conv2 and
        # conv1's resolutions - these mirror the 3 real pooling stages.
        self.upconv3 = self.expand_block(128, 64)
        self.upconv2 = self.expand_block(64 + 64, 32)
        self.upconv1 = self.expand_block(32 + 32, 32)

        # Final classification layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)  # H/2
        conv2 = self.conv2(conv1)  # H/4
        conv3 = self.conv3(conv2)  # H/8
        conv4 = self.conv4(conv3)  # H/8 (dilated, rate=2)
        conv5 = self.conv5(conv4)  # H/8 (dilated, rate=4)

        # Bottleneck
        bottleneck = self.bottleneck(conv5)  # H/8

        # Fuse stages - conv5/conv4/conv3 are all still at H/8, so these are
        # plain concatenate-and-convolve steps, no upsampling involved.
        upconv6 = self.upconv6(torch.cat([bottleneck, conv5], 1))
        upconv5 = self.upconv5(torch.cat([upconv6, conv4], 1))
        upconv4 = self.upconv4(torch.cat([upconv5, conv3], 1))

        # Expand stages - this is where spatial resolution is actually
        # recovered, matching the 3 real pooling stages in the encoder.
        upconv3 = self.upconv3(upconv4)
        upconv3 = self._match_size(upconv3, conv2)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))

        upconv2 = self._match_size(upconv2, conv1)
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        # Final output
        output = self.final_conv(upconv1)

        return output

    def _match_size(self, x, target):
        # Ensure x matches the spatial dimensions of target using interpolation.
        # With divisible input sizes this shouldn't fire at all in this design -
        # it's a safety net for odd input dimensions, not part of the normal path.
        if x.shape[2:] != target.shape[2:]:
            print("match size fired!")
            x = torch.nn.functional.interpolate(
                x,
                size=target.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        return x

    def contract_block(self, in_channels, out_channels):

        #Contracting block: 2 convolutions + batch norm + ReLU + dropout, then downsample with MaxPool

        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling by 2x
        )
        return contract

    def expand_block(self, in_channels, out_channels):

        #Expanding block: upsample, then 2 convolutions + batch norm + ReLU + dropout

        expand = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),  # Upsampling by 2x
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return expand

    def dilated_block(self, in_channels, out_channels, dilation):
        #
        # Dilated block: 2 dilated convolutions + batch norm + ReLU + dropout, no downsampling.
        # Grows the receptive field the way pooling would, without discarding resolution -
        # used at deeper stages to keep thin/small structures visible to the network.
        # padding=dilation keeps spatial size unchanged for a 3x3 kernel
        #
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def fuse_block(self, in_channels, out_channels):

        # Fuse block: 2 convolutions + batch norm + ReLU + dropout, no resolution change.
        # Same as contract_block minus the MaxPool - used to merge concatenated skip
        # connections that are already at the decoder's current resolution
        #
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block


#Attention + Dense blocks incorporated
#based on Dutt et al. Tidal creek segmentation architecture
#architecture trial code from LeeJunHyun/Image_segmentation github repo

from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Dropout2d(0.2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
