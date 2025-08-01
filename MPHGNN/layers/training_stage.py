import torch
import math
import torch.nn as nn
from itertools import chain
import torch.nn.functional as F
from torch.nn.modules.utils import _single
from MPHGNN.layers.train_model import CommonTorchTrainModel

class PReLU(nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.alpha = nn.parameter.Parameter(torch.empty(num_parameters, **factory_kwargs).fill_(init))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.prelu(input, self.alpha)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


class Lambda(nn.Module):
    def __init__(self, func) -> None:
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
        
def create_act(name=None):
    if name is None:
        return nn.Identity()
    elif name == "relu":
        return nn.ReLU()
    elif name == "prelu":
        return PReLU()
    elif name == "gelu":
        return GELU()
    elif name == "softmax":
        return nn.Softmax(dim=-1)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return Lambda(lambda x: x)
    else:
        raise Exception()


class Linear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

class SGConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, den=None, 
                 stride=1, padding=1, groups=1, dilation=1, bias=True):
        super().__init__()
        print("den value inside SGConv1d:", den)  
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups
        
        # Initialize learnable weights
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        
        # Create fixed weight pattern
        device = torch.device('cpu')
        alfa = torch.cat([
            torch.tensor(den, device=device),
            torch.tensor([1.0], device=device),
            torch.flip(torch.tensor(den, device=device), dims=[0])
        ])
        
        # Verify dimension matching
        if alfa.shape[0] != self.kernel_size[0]:
            raise ValueError(
                f"Alfa length ({alfa.shape[0]}) must match the kernel size ({self.kernel_size[0]})")
            
        self.register_buffer('alfa', alfa)  # Store as [K]
    
    def forward(self, x):
        # Ensure the same device as input
        alfa = self.alfa.to(x.device)
        
        # Reshape to [1, 1, K] for correct broadcasting
        alfa = alfa.view(1, 1, -1)
        weighted_weight = self.weight * alfa

        return F.conv1d(
            x, weighted_weight, self.bias, 
            self.stride, self.padding, self.dilation, self.groups)


class ECANet(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))   # Adaptively calculate kernel size
        kernel_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x)
        y = y.transpose(1, 2)  # [b, 1, c]
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(1, 2)  # [b, c, 1]
        return x * y  # # Recalibration

# Combining SGConv1d and ECANet
class SGConv1dWithECANet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=True, gamma=2, b=1, den=None):
        super().__init__()
        print(f"den value inside SGConv1dWithECANet: {den}")
        self.conv = SGConv1d(in_channels, out_channels, kernel_size, den, stride, padding, groups, dilation, bias)
        self.se = ECANet(out_channels, gamma=gamma, b=b)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x

class MLP(nn.Module):

    def __init__(self,
                 channels_list,
                 input_shape,
                 drop_rate=0.0,
                 activation=None,
                 output_drop_rate=0.0,
                 output_activation=None,
                 kernel_regularizer=None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.kernel_regularizer = kernel_regularizer

        in_channels = input_shape[-1]
        channels_list = [in_channels] + channels_list

        layers = [] 
        for i in range(len(channels_list) - 1):
            layers.append(Linear(channels_list[i], channels_list[i + 1]))
            if i < len(channels_list) - 2:
                layers.append(create_act(activation))
                layers.append(nn.Dropout(drop_rate))
            else:
                layers.append(create_act(output_activation))
                layers.append(nn.Dropout(output_drop_rate))


        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LocalSemanticEncoding(nn.Module):
    def __init__(self,
                 filters_list,
                 drop_rate,
                 input_shape,
                 kernel_regularizer=None,
                 den=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"den value inside LocalSemanticEncoding: {den}")
        self.hop_encoders = None
        self.filters_list = filters_list
        self.drop_rate = drop_rate
        self.kernel_regularizer = kernel_regularizer
        self.real_filters_list = None

        num_groups = len(input_shape)
        self.group_sizes = [group_shape[1] for group_shape in input_shape]
        self.real_filters_list = [self._get_real_filters(i) for i in range(num_groups)]
        
        self.group_encoders = nn.ModuleList([
            nn.Sequential(
                SGConv1dWithECANet(group_size, real_filters, kernel_size=3, den=den, padding=1),
                Lambda(lambda x: x.view(x.size(0), -1))
            )
            for _, (group_size, real_filters) in enumerate(zip(self.group_sizes, self.real_filters_list))
        ])

    def _get_real_filters(self, i):
        if self.group_sizes[i] == 1:
            return 1
        elif isinstance(self.filters_list, list):
            return self.filters_list[i]
        else:
            return self.filters_list
 

    def forward(self, x_group_list):
        group_h_list = []
        for i, (x_group, group_encoder) in enumerate(zip(x_group_list, self.group_encoders)):
            h = x_group
            group_h = group_encoder(h)
            group_h_list.append(group_h)
        return group_h_list



class GlobalSemanticFusion(nn.Module):
    def __init__(self,
                 group_channels_list,
                 global_channels_list,
                 merge_mode,
                 input_shape,
                 drop_rate=0.0,
                 activation="prelu",
                 output_activation=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.group_fc_list = None
        self.global_fc = None
        self.group_channels_list = group_channels_list
        self.global_channels_list = global_channels_list
        self.merge_mode = merge_mode
        self.drop_rate = drop_rate
        self.use_shared_group_fc = False
        self.group_encoder_mode = "common" 
        num_groups = len(input_shape)
        self.num_groups = num_groups

        self.group_fc_list = nn.ModuleList([
            MLP(
                group_channels_list, 
                input_shape=group_input_shape,
                drop_rate=drop_rate,
                activation=activation,
                output_drop_rate=drop_rate,
                output_activation=activation
            )
            for group_input_shape in input_shape
        ])

        #if merge_mode in ["mean", "free"]:
        if merge_mode == "mean":
            global_input_shape = [-1, group_channels_list[-1]]
        elif merge_mode == "concat":
            global_input_shape = [-1, group_channels_list[-1] * num_groups]
        else:
            raise Exception("wrong merge mode: ", merge_mode)

        self.global_fc = MLP(self.global_channels_list, 
                             input_shape=global_input_shape,
                             drop_rate=self.drop_rate, 
                             activation=activation,
                             output_drop_rate=0.0,
                             output_activation=output_activation)


    def forward(self, inputs):
        x_list = inputs
        group_h_list = [group_fc(x) for x, group_fc in zip(x_list, self.group_fc_list)]

        if self.merge_mode == "mean":
            global_h = torch.stack(group_h_list, dim=0).mean(dim=0)
        elif self.merge_mode == "concat":
            global_h = torch.concat(group_h_list, dim=-1)
        else:
            raise Exception("wrong merge mode: ", self.merge_mode)

        h = self.global_fc(global_h)
        return h

class FeatureEnhancement(nn.Module):
    def __init__(self, feature_dim, enhancement_ratio=0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.enhancement_dim = int(feature_dim * enhancement_ratio)
        self.enhancement_net = nn.Sequential(
            nn.Linear(feature_dim, self.enhancement_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.enhancement_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        enhancement = self.enhancement_net(x)
        return x * (1 + enhancement)

class MPHGNNEncoder(CommonTorchTrainModel):
    def __init__(self, filters_list, group_channels_list, global_channels_list, merge_mode, input_shape, *args,
                 input_drop_rate=0.0,
                 drop_rate=0.0,
                 activation="prelu",
                 output_activation=None,
                 den=None,
                 use_feature_enhancement=True,
                 **kwargs):

        super().__init__(*args, **kwargs)
        print(f"den value inside MPHGNNEncoder: {den}")
        self.input_dropout = nn.Dropout(input_drop_rate)
        self.input_drop_rate = input_drop_rate
        self.use_feature_enhancement = use_feature_enhancement
        local_semantic_input_shape = input_shape
        self.group_encoders = LocalSemanticEncoding(filters_list, drop_rate, local_semantic_input_shape, den=den)

        global_semantic_fusion_input_shape = [[-1, group_input_shape[-1] * filters]
                                          for group_input_shape, filters in zip(local_semantic_input_shape, self.group_encoders.real_filters_list)]
        self.global_semantic_fusion = GlobalSemanticFusion(
            group_channels_list, global_channels_list,
            merge_mode,
            input_shape=global_semantic_fusion_input_shape,
            drop_rate=drop_rate,
            activation=activation,
            output_activation=output_activation)

        final_dim = global_channels_list[-1]
        if self.use_feature_enhancement:
            self.feature_enhancement = FeatureEnhancement(final_dim)
        else:
            self.feature_enhancement = None

    def forward(self, inputs):
        x_group_list = inputs
        dropped_x_group_list = [F.dropout(x_group, self.input_drop_rate, training=self.training, inplace=False)
                                for x_group in x_group_list]

        h_list = self.group_encoders(dropped_x_group_list)
        h = self.global_semantic_fusion(h_list)
        if self.feature_enhancement:
            h = self.feature_enhancement(h)
        return h
