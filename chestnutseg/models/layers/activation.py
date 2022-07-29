import torch.nn as nn


class Activation(nn.Module):
    """
    The wrapper of activation.

    The code piece is excerpted from PaddlePaddle:
    https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/paddleseg/models/layers/activation.py

    Args:
        act (str, optional): The activation name in lowercase. Default: None,
            means identical transformation.
    
    Returns:
        A callable object of Activation.
    """
    def __init__(self, act=None):
        super(Activation, self).__init__()

        self._act = act
        upper_act_names = nn.modules.activation.__dict__.keys()
        lower_act_names = [act.lower() for act in upper_act_names]
        act_dict = dict(zip(lower_act_names, upper_act_names))
        
        if act is not None:
            if act in act_dict.keys():
                act_name = act_dict[act]
                self.act_func = eval("nn.{}()".format(act_name))
            else:
                raise KeyError("{} does not exist in the current {}".format(
                    act, act_dict.keys()))

    def forward(self, x):
        if self._act is not None:
            return self.act_func(x)
        else:
            return x
