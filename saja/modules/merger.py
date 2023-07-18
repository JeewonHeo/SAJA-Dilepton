import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple
from saja.data.utils import make_data_mask


class ParticleFlowMerger(nn.Module):
    def forward(self,
                jet: Tensor,
                jet_lengths: Tensor,
                jet_data_mask: Tensor,
                lepton: Tensor,
                met: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        jet = [j[m] for j, m in zip(jet, jet_data_mask)]

        # add a fake sequence dimension
        # (batch, met_features) -> (batch, 1, met_features)
        met = met.unsqueeze(1)

        particle_flow = zip(jet, lepton, met)
        particle_flow = [torch.cat(each, dim=0) for each in particle_flow]

        # lengths = [len(each) for each in particle_flow]
        lengths = jet_lengths + 2 + 1 # jet_lengths + lepton_lengths + met_lengths

        particle_flow = pad_sequence(particle_flow, batch_first=True)

        data_mask = make_data_mask(particle_flow, lengths)

        return particle_flow, lengths, data_mask

class ConParticleFlowMerger(nn.Module):
    def forward(self,
                track: Tensor,
                track_lengths: Tensor,
                track_data_mask: Tensor,
                tower: Tensor,
                tower_lengths: Tensor,
                tower_data_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        track = [t[m] for t, m in zip(track, track_data_mask)]
        tower = [t[m] for t, m in zip(tower, tower_data_mask)]

        particle_flow = zip(track, tower)
        particle_flow = [torch.cat(each, dim=0) for each in particle_flow]

        # lengths = [len(each) for each in particle_flow]
        lengths = track_lengths + tower_lengths # track_lengths + tower_lengths

        particle_flow = pad_sequence(particle_flow, batch_first=True)

        data_mask = make_data_mask(particle_flow, lengths)

        return particle_flow, lengths, data_mask
