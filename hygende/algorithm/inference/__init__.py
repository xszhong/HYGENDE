from ...register import Register

inference_register = Register("inference")

from .base import Inference
from .weight import WeightBasedInference
from .relevance import RelevanceBasedInference
from .random import RandomInference
