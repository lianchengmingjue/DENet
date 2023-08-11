from src.utils.model_init import *
from src.networks.resunet import SLBR


# our method
def slbr(**kwargs):
    return SLBR(args=kwargs['args'], shared_depth=1, blocks=3, long_skip=True)



