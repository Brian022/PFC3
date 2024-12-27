import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmrotate.models.builder import NECKS

@NECKS.register_module()
class DeNoisingFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 contrastive_dim=128,
                 start_level=0,
                 end_level=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(DeNoisingFPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = end_level
        self.contrastive_dim = contrastive_dim

        if end_level == -1:
            self.backbone_end_level = len(in_channels)
        else:
            self.backbone_end_level = end_level
        assert num_outs >= self.backbone_end_level - start_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # Encoders geométrico y semántico
        self.geometric_encoder = nn.Sequential(
            nn.Conv2d(out_channels, contrastive_dim, kernel_size=1),
            nn.BatchNorm2d(contrastive_dim),
            nn.ReLU(inplace=True)
        )
        self.semantic_encoder = nn.Sequential(
            nn.Conv2d(out_channels, contrastive_dim, kernel_size=1),
            nn.BatchNorm2d(contrastive_dim),
            nn.ReLU(inplace=True)
        )

        # Parámetros para ajuste dinámico del ruido
        self.noise_weights = nn.Parameter(torch.ones(len(self.lateral_convs)))

    def contrastive_loss(self, geometric_features, semantic_features):
        """Cálculo de la pérdida contrastiva."""
        # Normalización
        geo_norm = geometric_features / geometric_features.norm(dim=1, keepdim=True)
        sem_norm = semantic_features / semantic_features.norm(dim=1, keepdim=True)
        # Similaridad con producto punto
        similarity = torch.mm(geo_norm, sem_norm.t())
        # Pérdida contrastiva básica
        loss = -torch.log(torch.diag(similarity) / similarity.sum(dim=1))
        return loss.mean()

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # Construcción de laterales
        laterals = [
            lateral_conv(inputs[i]) * self.noise_weights[i]
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Construcción del top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')

        # Aplicar encoders geométrico y semántico
        geometric_features = [self.geometric_encoder(feat) for feat in laterals]
        semantic_features = [self.semantic_encoder(feat) for feat in laterals]

        # Calcular pérdida contrastiva en modo entrenamiento
        if self.training:
            loss_contrastive = self.contrastive_loss(
                torch.cat(geometric_features, dim=0),
                torch.cat(semantic_features, dim=0)
            )
            return laterals, loss_contrastive

        # Generar salidas de la FPN
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        return tuple(outs)


