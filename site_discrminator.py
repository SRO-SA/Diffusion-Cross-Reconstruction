
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function


class DomainDiscriminator(nn.Sequential):
    """
    Adapted from https://github.com/thuml/Transfer-Learning-Library

    Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_
    In the original paper and implementation, we distinguish whether the input features come
    from the source domain or the target domain.

    We extended this to work with multiple domains, which is controlled by the n_domains
    argument.

    Args:
        in_feature (int): dimension of the input feature
        n_domains (int): number of domains to discriminate
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.
    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, n_domains)`
    """

    def __init__(
        self, in_feature: int, n_domains, hidden_size: int = 1024, batch_norm=True
    ):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_domains),
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, n_domains),
            )

    def get_parameters_with_lr(self, lr) -> List[Dict]:
        return [{"params": self.parameters(), "lr": lr}]

class GradientReverseFunction(Function):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.0
    ) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class DomainAdversarialNetwork(nn.Module):
    def __init__(self, featurizer, classifier, n_domains):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        print("featurizer domain: ", featurizer.d_out, "number of domains: ", n_domains)
        print("batch norm: ", featurizer.batch_norm)
        self.domain_classifier = DomainDiscriminator(featurizer.d_out, n_domains, batch_norm=featurizer.batch_norm)
        self.gradient_reverse_layer = GradientReverseLayer()

    def forward(self, input):
        features = self.featurizer(input)
        y_pred = self.classifier(features)
        features = self.gradient_reverse_layer(features)
        # print(features.shape)
        domains_pred = self.domain_classifier(features)
        return y_pred, domains_pred

    def get_parameters_with_lr(self, featurizer_lr, classifier_lr, discriminator_lr) -> List[Dict]:
        """
        Adapted from https://github.com/thuml/Transfer-Learning-Library

        A parameter list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer
        """
        # In TLL's implementation, the learning rate of this classifier is set 10 times to that of the
        # feature extractor for better accuracy by default. For our implementation, we allow the learning
        # rates to be passed in separately for featurizer and classifier.
        params = [
            {"params": self.featurizer.parameters(), "lr": featurizer_lr},
            {"params": self.classifier.parameters(), "lr": classifier_lr},
        ]
        return params + self.domain_classifier.get_parameters_with_lr(discriminator_lr)



# --- add at top of server.py -----------------------------------------------
class ParamDiscriminator(torch.nn.Module):
    """Takes a *flattened* parameter vector and predicts the domain ID."""
    def __init__(self, d_in, n_domains, hidden=1024):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_domains)
        )

    def forward(self, θ_flat):
        return self.net(θ_flat)
# ---------------------------------------------------------------------------


""" 3D brain age model with separate feature extractor and regressor"""
# mean absolute error
# adam optimizer
# learning rate 10^-4
# weight decay 10^-4
from torch import nn
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

def conv_blk(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_channel), nn.MaxPool3d(2, stride=2), nn.ReLU()
    )



# ------------------ 1. Featurizer ------------------ #
class BrainCancerFeaturizer(nn.Module):
    """Return the feature *map* produced by conv4 or conv5
       + a global-pooled vector version (d_out = 256)."""
    def __init__(self, use_conv5: bool = True, d_out: int = 256):
        super().__init__()
        # copy/paste from your original model
        self.conv1 = conv_blk(1, 32)
        self.conv2 = conv_blk(32, 64)
        self.conv3 = conv_blk(64, 128)
        self.conv4 = conv_blk(128, 256)
        self.conv5 = conv_blk(256, 256)           # ← optional
        self.use_conv5 = use_conv5

        # global pooling produces (B,256,1,1,1) → (B,256)
        # self.gap = nn.AdaptiveAvgPool3d(1)
        self.d_out = d_out * 3*4*3                       # <- for DomainDiscriminator
        self.batch_norm = False                   # matches InstanceNorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, *x.shape[-3:])
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if self.use_conv5:
            x = self.conv5(x)                     # features after conv5
        return x                                  # (B,256,D',H',W')

    def map_to_vec(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.gap(fmap).flatten(1)          # (B,256)


# ------------------ 2. Regressor head ------------------ #
class BrainCancerRegressor(nn.Module):
    """Takes the feature map produced above and outputs the age scalar."""
    def __init__(self):
        super().__init__()
        self.conv6 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1, stride=1),
            nn.InstanceNorm3d(64), nn.ReLU(),
            nn.AvgPool3d(kernel_size=(2, 3, 2))
        )
        self.drop   = nn.Identity()
        self.output = nn.Conv3d(64, 1, kernel_size=1, stride=1)
        nn.init.constant_(self.output.bias, 62.68)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        x = self.conv6(fmap)
        x = self.drop(x)
        x = self.output(x)          # [B,1,D,H,W]
        x = x.mean(dim=(2,3,4))     # [B,1]  global average over space
        # print("after output:", x.shape)  # <-- this will reveal [B,1,1,1,2] etc.
        return x[:, 0]              # [B]


# ------------------ 4. Downstream age + gender model ------------------ #
class AgeGender3D(nn.Module):
    """
    Shared 3D featurizer, with:
      - age head  (same structure as BrainCancerRegressor)
      - gender head (2-class classifier)
    Forward:
        x -> (age_pred [B], gender_logits [B,2])
    """
    def __init__(self, featurizer: BrainCancerFeaturizer):
        super().__init__()
        self.featurizer = featurizer
        self.age_head = BrainCancerRegressor()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            age_pred:      [B]   (regression)
            gender_logits: [B,2] (classification)
        """
        fmap = self.featurizer(x)       # [B,256,D',H',W']
        age_pred = self.age_head(fmap)  # [B]
        return age_pred


# ------------------ 3. Full DANN wrapper ------------------ #
class DANN3D(nn.Module):
    """Domain-Adversarial Net for 3-D volumes – plugs straight into your trainer."""
    def __init__(self,
                 featurizer: BrainCancerFeaturizer,
                 n_domains : int,
                 hidden_size: int = 1024):
        super().__init__()
        self.featurizer = featurizer
        # self.regressor  = regressor
        # self.grl        = GradientReverseLayer()      # from your snippet
        # self.vec_pool   = nn.AdaptiveAvgPool3d(1)

        # Domain discriminator now expects a 256-D vector
        in_feat = self.featurizer.d_out                                     # ← discriminator input
        self.domain_classifier = DomainDiscriminator(
            in_feature = in_feat,
            n_domains  = n_domains,
            hidden_size= hidden_size,
            batch_norm = featurizer.batch_norm
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fmap      = self.featurizer(x)               # (B,256,D',H',W')
        # print("feature map shape:", fmap.shape)
        # y_pred    = self.regressor(fmap)             # (B,)
        feat_vec  = fmap.flatten(1)   # (B,256)
        dom_pred  = self.domain_classifier(feat_vec)
        return dom_pred
