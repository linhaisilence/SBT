import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

__all__ = ['ResNet', 'resnet50', 'resnet101']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, stride, stride),
                               bias=False)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=(1, 1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv4 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = out + residual
        return self.relu(out)


class ResNet(nn.Module):

    def __init__(self, block, layers, dim, depth, num_frames, heads=16, dim_head=64, scale_dim=4, dropout=0.,
                 pool='cls', num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        image_height, image_width = pair(28)
        patch_height, patch_width = pair(7)
        patch_dim = 128 * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c t (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.to_patch_dim = nn.Sequential(
            nn.Conv3d(512 * block.expansion, 128 * block.expansion, 1, 1, 0, bias=False),
            nn.BatchNorm3d(128 * block.expansion)
        )
        self.to_patch_embedding_global = nn.Sequential(
            Rearrange('b c t (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim,dim),
            nn.LayerNorm(dim)
        )
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 2, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        ti = x
        gl = x
        gl = self.layer3(gl)
        gl = self.layer4(gl)

        gl = self.to_patch_dim(gl)  # b t c p1 p2
        gl = self.to_patch_embedding_global(gl)  # b t 1 (p1 p2 c)
        ti = self.to_patch_embedding(ti)  # b t (h w) (p1 p2 c)
        ti = torch.cat((ti, gl), dim=2)  # b t (h w)+1 (p1 p2 c)
        b, t, n, _ = ti.shape
        # print(ti.size())
        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        ti = torch.cat((cls_space_tokens, ti), dim=2)
        to = self.pos_embedding[:, :, :(n + 1)] + ti
        to = self.dropout(to)
        to = rearrange(to, 'b n t d -> (b t) n d')
        s_out = self.space_transformer(to)
        s_out = rearrange(s_out[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        s_out = torch.cat((cls_temporal_tokens, s_out), dim=1)

        t_out = self.temporal_transformer(s_out)

        t_out = t_out.mean(dim=1) if self.pool == 'mean' else t_out[:, 0]
        # print(t_out.size())
        return t_out
        # x = x.transpose(1, 2).contiguous()
        # x = x.view((-1,) + x.size()[2:])
        #
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)


def resnet50(dim, depth, num_frames, **kwargs):
    """Constructs a ResNet-50 based model.
	"""
    model = ResNet(Bottleneck, [3, 4, 2, 2], dim, depth, num_frames, **kwargs)
    # checkpoint = model_zoo.load_url(model_urls['resnet50'])
    # checkpoint = torch.load("D:/pythonProject1/checkpoints/resnet50-19c8e357.pth")
    # layer_name = list(checkpoint.keys())
    # for ln in layer_name:
    #     if 'conv' in ln or 'downsample.0.weight' in ln:
    #         checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    #     if 'conv2' in ln:
    #         n_out, n_in, _, _, _ = checkpoint[ln].size()
    #         checkpoint[ln] = checkpoint[ln][:n_out // alpha * (alpha - 1), :n_in // beta, :, :, :]
    # model.load_state_dict(checkpoint, strict=False)

    return model


def resnet101(alpha, beta, **kwargs):
    """Constructs a ResNet-101 model.
	Args:
		groups
	"""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet101'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
        if 'conv2' in ln:
            n_out, n_in, _, _, _ = checkpoint[ln].size()
            checkpoint[ln] = checkpoint[ln][:n_out // alpha * (alpha - 1), :n_in // beta, :, :, :]
    model.load_state_dict(checkpoint, strict=False)

    return model
