# Copyright 2021 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Contrastive learning specialized losses.
"""
from .barlow import Barlow  # noqa
from .circle_loss import CircleLoss  # noqa
from .metric_loss import MetricLoss  # noqa
from .multisim_loss import MultiSimilarityLoss  # noqa
from .pn_loss import PNLoss  # noqa
from .simclr import SimCLRLoss  # noqa
from .simsiam import SimSiamLoss  # noqa
from .softnn_loss import SoftNearestNeighborLoss  # noqa
from .triplet_loss import TripletLoss  # noqa
from .vicreg import VicReg  # noqa
