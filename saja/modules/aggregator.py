import torch
from torch import Tensor
import torch.nn as nn

@torch.jit.script
def make_scatter_add_index(lengths: Tensor,
                           num_features: int,
) -> Tensor:
    r"""
    Example::
      >>> make_scatter_add_index(lengths=torch.tensor([2, 1, 3]))
      torch.tensor([0, 0, 1, 2, 2, 2])
    """
    index = [lengths.new_full(size=[int(each), ], fill_value=idx) for idx, each in enumerate(lengths)]
    index = torch.cat(index)
    index = index.unsqueeze(1).repeat(1, num_features)
    return index

class ScatterMean(nn.Module):
    def forward(self,
                input: Tensor,
                data_mask: Tensor,
                lengths: Tensor) -> Tensor:
        """scatter mean
        """
        batch_size, _, num_features = input.shape

        data_mask = data_mask.unsqueeze(2)
        input = input.masked_select(data_mask)
        input = input.reshape(-1, num_features)
        index = self.make_scatter_add_index(lengths, num_features)
        output = torch.scatter_add(
            input=input.new_zeros((batch_size, num_features)),
            dim=0,
            index=index,
            src=input)
        output = output / lengths.unsqueeze(1).to(output.dtype)
        return output


    @staticmethod
    def make_scatter_add_index(lengths: Tensor,
                               num_features: int,
    ) -> Tensor:
        return make_scatter_add_index(lengths=lengths,
                                      num_features=num_features)

