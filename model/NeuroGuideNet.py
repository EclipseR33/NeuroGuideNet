from utils import CenterLoss

import torch
import torch.nn as nn
from torch.nn import init
import math


class Conv2DMultiKernel(nn.Module):
    def __init__(self, in_ch, h_ch, conv_kernels, dropout,
                 channel_kernel=None, groups=1,
                 use_activation=True, use_dropout=True):
        super().__init__()
        assert h_ch % len(conv_kernels) == 0

        self.use_activation = use_activation
        self.use_dropout = use_dropout

        padding_list = [(0, kernel // 2) for kernel in conv_kernels]
        if channel_kernel is not None:
            conv_kernels = [(channel_kernel, kernel) for kernel in conv_kernels]
        else:
            conv_kernels = [(1, kernel) for kernel in conv_kernels]
        self.conv_list = nn.ModuleList([
            nn.Conv2d(in_ch, h_ch // len(conv_kernels), kernel, padding=padding, groups=groups)
            for kernel, padding in zip(conv_kernels, padding_list)
        ])
        self.bn = nn.BatchNorm2d(h_ch)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output_list = []
        for conv in self.conv_list:
            output_list.append(conv(x))
        output = torch.cat(output_list, dim=1)
        output = self.bn(output)
        if self.use_activation:
            output = self.elu(output)
        if self.use_dropout:
            output = self.dropout(output)
        return output


class Conv1DMultiKernel(nn.Module):
    def __init__(self, in_ch, h_ch, conv_kernels, dropout, groups=1,
                 use_activation=True, use_dropout=True):
        super().__init__()
        assert h_ch % len(conv_kernels) == 0

        self.use_activation = use_activation
        self.use_dropout = use_dropout

        padding_list = [kernel // 2 for kernel in conv_kernels]
        conv_kernels = [kernel for kernel in conv_kernels]
        self.conv_list = nn.ModuleList([
            nn.Conv1d(in_ch, h_ch // len(conv_kernels), kernel, groups=groups, padding=padding)
            for kernel, padding in zip(conv_kernels, padding_list)
        ])
        self.bn = nn.BatchNorm1d(h_ch)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output_list = []
        for conv in self.conv_list:
            output_list.append(conv(x))
        output = torch.cat(output_list, dim=1)
        output = self.bn(output)
        if self.use_activation:
            output = self.elu(output)
        if self.use_dropout:
            output = self.dropout(output)
        return output


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, output_attention=False):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads

        self.output_attention = output_attention

        self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)

        self.out_projection = nn.Linear(self.d_values * n_heads, d_model)
        self.value_projection = nn.Linear(d_model, self.d_values * n_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, get_graph=False):
        B, L, _ = x.shape
        H = self.n_heads

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, L, H, -1)
        values = self.value_projection(x).view(B, L, H, -1)

        E = queries.shape[-1]

        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        V = torch.einsum("bhls,bshd->blhd", A, values)

        V = self.out_projection(V.reshape(B, L, -1))
        if get_graph:
            return V, torch.softmax(scale * scores, dim=-1)
        else:
            return V


class CrossRegionFC_Block(nn.Module):
    def __init__(self, in_ch, d_model, pool_size, conv_kernels,
                 n_heads, ch_size, dropout):
        super(CrossRegionFC_Block, self).__init__()
        self.d_model = d_model
        self.ch_size = ch_size

        self.T_conv = Conv1DMultiKernel(in_ch, d_model, conv_kernels, dropout)
        self.T_conv_avg = Conv1DMultiKernel(in_ch, d_model, conv_kernels, dropout)
        self.res_conv = nn.Conv2d(in_ch, d_model, 1) \
            if in_ch != d_model else nn.Identity()
        self.ST_avg_pool = nn.AvgPool1d(pool_size)

        self.spatial_attention = AttentionLayer(d_model, n_heads, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, dim, channels, seq_len = x.shape
        conv_input = x.permute(0, 2, 1, 3)
        conv_input = conv_input.reshape(-1, dim, seq_len)

        conv_output = self.T_conv(conv_input) + self.res_conv(conv_input)
        conv_output = self.T_avg_pool(conv_output)

        conv_output = conv_output.reshape(batch_size, channels, self.d_model, -1)
        x = conv_output.permute(0, 2, 1, 3)

        # Spatial Attention
        batch_size, dim, channels, seq_len = x.shape
        attn_input = x.permute(0, 3, 2, 1)
        attn_input = attn_input.reshape(-1, channels, dim)

        attn_out = self.dropout(self.spatial_attention(attn_input)) + attn_input

        attn_out = attn_out.reshape(batch_size, seq_len, channels, dim)
        attn_out = attn_out.permute(0, 3, 2, 1)

        return attn_out


class CrossRegionFC(nn.Module):
    def __init__(self, in_ch: int, block_dims: list[int], pool_size: list[int],
                 block_kernels, n_heads_list: list[int], ch_size, dropout):
        # Cross-region Feature Collaboration
        super(CrossRegionFC, self).__init__()
        self.ch_size = ch_size

        block_in_ch = [in_ch] + block_dims[:-1]
        self.block = nn.ModuleList([
            CrossRegionFC_Block(
                block_in, d_model, p_size, block_kernels,
                n_heads, self.ch_size, dropout,
            )
            for i, (block_in, d_model, p_size, n_heads) in enumerate(
                zip(block_in_ch, block_dims, pool_size, n_heads_list)
            )
        ])

        self.channel_fuse = Conv2DMultiKernel(
            block_dims[-1], block_dims[-1], [1], dropout,
            channel_kernel=self.ch_size, groups=block_dims[-1]
        )

    def forward(self, x):
        cur_x = x
        for block in self.block:
            cur_out = block(cur_x)
            cur_x = cur_out

        cur_x = self.channel_fuse(cur_x)
        return cur_x


class STBlock(nn.Module):
    def __init__(self, in_ch: int, d_model: int, pool_size, conv_kernels,
                 ch_size, dropout, use_channel_fuse=False):
        super(STBlock, self).__init__()
        self.d_model = d_model
        self.ch_size = ch_size
        self.use_channel_fuse = use_channel_fuse
        temporal_in_ch = in_ch
        if use_channel_fuse:
            self.s_conv = Conv2DMultiKernel(in_ch, d_model, conv_kernels, dropout,
                                                       channel_kernel=ch_size,
                                                       groups=in_ch // len(conv_kernels))
            temporal_in_ch = d_model

        self.t_conv = Conv2DMultiKernel(temporal_in_ch, d_model, conv_kernels, dropout,
                                               groups=temporal_in_ch // len(conv_kernels))
        self.res_conv = nn.Conv2d(temporal_in_ch, d_model, 1) \
            if temporal_in_ch != d_model else nn.Identity()

        self.avg_pool = nn.AvgPool2d((1, pool_size))

    def forward(self, x):
        if self.use_channel_fuse:
            x = self.s_conv(x)

        x = self.t_conv(x) + self.res_conv(x)

        x = self.avg_pool(x)
        return x


class RegionSTModule(nn.Module):
    def __init__(self, in_ch: int, block_dims: list[int], pool_size: list[int],
                 block_kernels, ch_size,
                 dropout, channel_fuse_idx=0):
        # Region-specific Spatiotemporal Feature Extraction
        super(RegionSTModule, self).__init__()
        self.ch_size = ch_size

        block_in_ch = [in_ch] + block_dims[:-1]
        self.block = nn.ModuleList([
            STBlock(
                in_ch=block_in, d_model=d_model, pool_size=p_size,
                conv_kernels=block_kernels,
                ch_size=self.ch_size, dropout=dropout,
                use_channel_fuse=True if i == channel_fuse_idx else False
            )
            for i, (block_in, d_model, p_size) in enumerate(
                zip(block_in_ch, block_dims, pool_size)
            )
        ])

    def forward(self, x):
        cur_x = x
        for block in self.block:
            cur_out = block(cur_x)
            cur_x = cur_out

        return cur_x


def get_bciiv2a_channel_parcellation():
    bci_ch_name = [
        'Fz',
        'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2',
        'POz'
    ]
    bci_ch_name_parcellation = [
        # C
        'C3', 'C1', 'Cz', 'C2', 'C4',
        # L half
        'FC3', 'FC1', 'C5', 'C3', 'C1', 'CP3', 'CP1', 'P1',
        # R half
        'FC2', 'FC4', 'C2', 'C4', 'C6', 'CP2', 'CP4', 'P2',
        # F
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        # LC
        'C5', 'C3', 'C1', 'Cz', 'CP3', 'CP1', 'CPz',
        # RC
        'Cz', 'C2', 'C4', 'C6', 'CPz', 'CP2', 'CP4',
        # P
        'P1', 'Pz', 'P2', 'POz'
    ]
    ch_idx_parcellation = [bci_ch_name.index(name) for name in bci_ch_name_parcellation]
    ch_parcellation_size = [5, 8, 8, 6, 7, 7, 4]
    return ch_idx_parcellation, ch_parcellation_size


def get_OpenBMI_channel_parcellation():
    OpenBMI_ch_name = [
        'Fp1', 'Fp2',
        'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'PO9', 'O1', 'Oz', 'O2', 'PO10',
        'FC3', 'FC4',
        'C5', 'C1', 'C2', 'C6',
        'CP3', 'CPz', 'CP4',
        'P1', 'P2',
        'POz',
        'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h',
        'TPP8h', 'TP8', 'TPP10h',
        'F9', 'F10',
        'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'
    ]
    OpenBMI_ch_name_parcellation = [
        # C
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        # F L
        'Fp1', 'AF7', 'AF3', 'F9', 'F7', 'F3', 'Fz', 'FT9', 'FC5', 'FC3', 'FC1',
        # F R
        'Fp2', 'AF4', 'AF8', 'Fz', 'F4', 'F8', 'F10', 'FC2', 'FC4', 'FC6', 'FT10',
        # C L
        'FTT9h', 'T7', 'C5', 'C3', 'C1', 'Cz', 'TTP7h', 'TP9', 'TP7', 'CP5', 'CP3',
        'CP1', 'CPz', 'TPP9h',
        # C R
        'FTT10h', 'Cz', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
        'TP10', 'TPP8h', 'TPP10h',
        # P L
        'P7', 'P3', 'P1', 'Pz', 'PO9', 'PO3', 'POz', 'O1', 'Oz',
        # P R
        'Pz', 'P2', 'P4', 'P8', 'POz', 'PO4', 'PO10', 'Oz', 'O2',
        # F
        'Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'Fz', 'F4', 'FC3', 'FC1', 'FC2', 'FC4',
        # T L
        'FT9', 'FTT9h', 'T7', 'TTP7h', 'TP9', 'TP7', 'TPP9h',
        # T R
        'FT10', 'FTT10h', 'T8', 'TP8', 'TP10', 'TPP8h', 'TPP10h',
        # O
        'PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2',
        # L half
        'Fp1', 'AF7', 'AF3', 'F9', 'F7', 'F3', 'FT9', 'FC5', 'FC3', 'FC1', 'FTT9h',
        'T7', 'C5', 'C3', 'C1', 'TTP7h', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'TPP9h',
        'P7', 'P3', 'P1', 'PO9', 'PO3', 'O1',
        # R half
        'Fp2', 'AF4', 'AF8', 'F4', 'F8', 'F10', 'FC2', 'FC4', 'FC6', 'FT10', 'FTT10h',
        'C2', 'C4', 'C6', 'T8', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'TPP8h', 'TPP10h',
        'P2', 'P4', 'P8', 'PO4', 'PO10', 'O2'
    ]
    ch_idx_parcellation = [OpenBMI_ch_name.index(name) for name in OpenBMI_ch_name_parcellation]
    ch_parcellation_size = [7, 11, 11, 14, 14, 9, 9, 11, 7, 7, 6, 28, 28]
    return ch_idx_parcellation, ch_parcellation_size


class LocalRegionSTModule(nn.Module):
    def __init__(self, module_cfg, ds_name):
        """
            Functionally-Guided Hierarchical EEG Channel Parcellation
            &
            Region-specific Spatiotemporal Feature Extraction
        """
        super().__init__()

        # Channel Parcellation
        if ds_name == 'BCIIV2a':
            self.ch_idx_parcellation, self.ch_parcellation_size = get_bciiv2a_channel_parcellation()
        elif ds_name == 'OpenBMI':
            self.ch_idx_parcellation, self.ch_parcellation_size = get_OpenBMI_channel_parcellation()
        else:
            raise 'Unsupported dataset'

        self.RST_module = nn.ModuleList(
            [
                RegionSTModule(
                    in_ch=module_cfg['block_dims'][0],
                    block_dims=module_cfg['block_dims'],
                    pool_size=module_cfg['pool_size'],
                    block_kernels=module_cfg['block_kernels'],
                    ch_size=ch_size,
                    dropout=module_cfg['dropout'],
                    channel_fuse_idx=module_cfg['channel_fuse_idx']
                )
                for ch_size in self.ch_parcellation_size
            ]
        )

    def forward(self, x):
        output_list = []
        start_idx = 0
        for i, ch_size in enumerate(self.ch_parcellation_size):
            tensor_index = self.ch_idx_parcellation[start_idx:start_idx + ch_size]
            parcellation_output = self.RST_module[i](x[:, :, tensor_index])
            output_list.append(parcellation_output)
            start_idx += ch_size
        output = torch.cat(output_list, dim=2)
        return output


class NeuroGuideNet(nn.Module):
    def __init__(self, model_cfg, ds_name):
        super().__init__()
        self.n_channels = model_cfg['n_channels']
        self.num_class = model_cfg['num_class']
        self.in_channels = model_cfg['in_channels']
        self.use_input_conv = model_cfg['use_input_conv']
        self.in_kernel = model_cfg['in_kernel']
        self.use_conv_fuse = model_cfg['use_conv_fuse']
        self.emb_size = model_cfg['cross_region_fc_module']['block_dims'][0]
        self.mi_feature_len = model_cfg['cross_region_fc_module']['block_dims'][-1]

        self.out_conv_ch = model_cfg['localRST_module']['block_dims'][0]
        self.input_conv = nn.Identity()
        if self.use_input_conv:
            if isinstance(self.in_kernel, int):
                self.input_conv = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_conv_ch,
                              kernel_size=(1, self.in_kernel),
                              padding=(0, self.in_kernel // 2),
                              groups=1),
                    nn.BatchNorm2d(model_cfg['localRST_module']['block_dims'][0])
                )
            else:
                self.input_conv = Conv2DMultiKernel(
                    in_ch=self.in_channels,
                    h_ch=self.out_conv_ch,
                    conv_kernels=self.in_kernel,
                    dropout=0,
                    use_activation=False,
                    use_dropout=False
                )

        gs_cfg = model_cfg['globalRST_module']
        self.localRST_module = LocalRegionSTModule(model_cfg['localRST_module'], ds_name)
        self.globalRST_module = RegionSTModule(
            in_ch=gs_cfg['block_dims'][0],
            block_dims=gs_cfg['block_dims'],
            pool_size=gs_cfg['pool_size'],
            block_kernels=gs_cfg['block_kernels'],
            ch_size=gs_cfg['ch_size'],
            dropout=gs_cfg['dropout'],
            channel_fuse_idx=gs_cfg['channel_fuse_idx']
        )

        # Globally-guided Cross-region Feature Collaboration and Aggregation
        self.aggregation_conv = nn.Conv2d(self.emb_size, self.emb_size * 2, kernel_size=1, padding=0)

        crfc_cfg = model_cfg['cross_region_fc_module']
        self.cross_region_fc_module = CrossRegionFC(
            in_ch=self.emb_size,
            block_dims=crfc_cfg['block_dims'],
            pool_size=crfc_cfg['pool_size'],
            block_kernels=crfc_cfg['block_kernels'],
            n_heads_list=crfc_cfg['n_heads_list'],
            ch_size=crfc_cfg['ch_size'],
            dropout=crfc_cfg['dropout']
        )

        self.adaptive_spatial_conv = nn.Conv2d(
            in_channels=self.emb_size,
            out_channels=self.emb_size,
            kernel_size=(len(self.localRST_module.ch_parcellation_size), 1),
            stride=(1, 1),
            groups=self.emb_size
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier_mi = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.mi_feature_len, self.num_class)
        )
        self.center_loss = CenterLoss(model_cfg['num_class'],
                                      self.emb_size,
                                      device=model_cfg['device'])
        self.init_params()

    def init_params(self):
        init.zeros_(self.aggregation_conv.weight)
        init.zeros_(self.aggregation_conv.bias)

    def get_center_loss(self, x, label):
        return self.center_loss(x, label)

    def spatial_fusion(self, local_feature, global_feature):
        scale, shift = self.aggregation_conv(global_feature).chunk(2, dim=1)
        fusion_output = local_feature * (1 + scale) + shift
        return fusion_output

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        if self.input_conv is not None:
            x = self.input_conv(x)

        local_output = self.localRST_module(x)
        global_spatial_output = self.globalRST_module(x)
        spatial_output = self.spatial_fusion(local_output, global_spatial_output)

        ST_output = self.cross_region_fc_module(spatial_output)
        if self.use_conv_fuse:
            ST_output = self.adaptive_spatial_conv(ST_output)

        mi_emb = ST_output.squeeze(dim=2)
        mi_emb = self.avg_pool(mi_emb)
        predict = self.classifier_mi(mi_emb)

        output_dict = {'mi_emb': mi_emb, 'subject_emb': None}
        return predict, output_dict
