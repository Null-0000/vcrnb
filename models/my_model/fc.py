from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from torch import nn
from allennlp.nn import Activation


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            input_dropout: float = None,
            output_dropout: float = None
    ):
        super().__init__()

        layers = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        if input_dropout is not None:
            layers.append(nn.Dropout(p=input_dropout))
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.LeakyReLU(negative_slope=0.1))
        if output_dropout is not None:
            layers.append(nn.Dropout(p=output_dropout))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
