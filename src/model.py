import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)
        
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, 3 * (num_classes + 5), bn_act=False, kernel_size=1)
            ## Cada célula possui 3 ancoras, cada ancora possui a probabilidade de ser uma classe
            ## e uma bounding box com a probabilidade de ter um objeto + x, y, w, h
        )
        self.num_classes = num_classes

    def forward(self, x):
        batch_size, anchor_per_cel = x.shape[0], 3
        feat_per_anchor = self.num_classes + 5
        w, h = x.shape[2], x.shape[3]
        return (
            self.pred(x)
            .reshape(batch_size, anchor_per_cel, feat_per_anchor, w, h)
            .permute(0, 1, 3, 4, 2)
        ) ## (batch_size, anchor, w, h, feats)
        
class SimpleFeaturePyramid(nn.Module):
    def __init__(self, in_channel, out_channels) -> None:
        super().__init__()

        self.first_stage = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(in_channel // 2, in_channel // 4, kernel_size=2, stride=2),
            nn.Conv2d(in_channel // 4, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        self.second_stage = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2),

            nn.Conv2d(in_channel // 2, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        self.third_stage = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        pred = []

        pred.append(self.first_stage(x))
        pred.append(self.second_stage(x))
        pred.append(self.third_stage(x))

        return pred


class DinoYolo(nn.Module):
    def __init__(self, dino_model, num_classes=20) -> None:
        super().__init__()

        self.dino_model = dino_model
        self.num_classes = num_classes
        self.transpose_conv = nn.ConvTranspose2d(6, 12, kernel_size=3, stride=2, padding=3)
        self.conv_block = nn.Sequential(
            CNNBlock(12, 24, kernel_size=3, stride=1, padding=1),
            CNNBlock(24, 48, kernel_size=3, stride=1, padding=1),
            CNNBlock(48, 96, kernel_size=3, stride=1, padding=1)
            
        )
        self.pyramid = SimpleFeaturePyramid(96, 256)
        self.detect = ScalePrediction(256, num_classes)

    def forward(self, x):
        outputs = []

        x = self.dino_model(x)
        x = x.reshape(-1, 6, 8, 8)
        x = self.transpose_conv(x)
        print(f"Shape is {x.shape}")
        x = self.conv_block(x)

        stages = self.pyramid(x)

        for stage in stages:
            outputs.append(self.detect(stage))

        return outputs

num_classes = 20
IMAGE_SIZE = 224
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model = DinoYolo(dinov2_vits14, 20)

x = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
out = model.forward(x)

assert out[0].shape == (1, 3, 80, 80, num_classes+5), out[0].shape
assert out[0].shape == (1, 3, 40, 40, num_classes+5)
assert out[0].shape == (1, 3, 20, 20, num_classes+5)

print("Deu bom")







