from typing import Tuple
import torch
from torch import Tensor
from torch import nn

from saja.modules import SelfAttentionBlock
from saja.modules import DecoderBlock
from saja.modules import ObjWise
from saja.modules import ParticleFlowMerger
from saja.modules import ConParticleFlowMerger
from saja.modules import ScatterMean

from saja.data.dataset import TTbarDileptonBatch
from saja.data.dataset import TTDileptonWithConBatch
from saja.data.dataset import ConBatch

from torch.nn.utils.rnn import pad_sequence


class SaJa(nn.Module):
    def __init__(self,
                 dim_input: int,
                 dim_output: int,
                 dim_ffn: int = 1024,
                 num_blocks: int = 6,
                 num_heads: int = 10,
                 depth: int = 32,
                 dropout_rate: float = 0.1,
    ) -> None:
        """
        """
        super(SaJa, self).__init__()
        self.dim_input = dim_input
        self.dim_ffn = dim_ffn
        self.num_blocks = num_blocks
        self.depth = depth
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.dim_output = dim_output

        self.dim_model = num_heads * depth

        self.ffn_bottom = ObjWise(
            nn.Linear(dim_input, dim_ffn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffn, self.dim_model, bias=True),
            nn.LeakyReLU())

        attention_blocks = []
        for _ in range(num_blocks):
            block = SelfAttentionBlock(self.dim_model, num_heads, dim_ffn,
                                       dropout_rate)
            attention_blocks.append(block)
        self.attention_blocks = nn.ModuleList(attention_blocks)

        self.ffn_top = ObjWise(
            nn.Linear(self.dim_model, dim_ffn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffn, dim_output, bias=True))

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:
            mask:
        Returns:
        """
        x = self.ffn_bottom(x, mask)
        attention_list = []
        for block in self.attention_blocks:
            x, attention = block(x, mask)
            attention_list.append(attention)
        x = self.ffn_top(x, mask)
        attention_list = torch.stack(attention_list, dim=1)
        return (x, attention_list)


class TTbarDileptonSAJA(nn.Module):
    def __init__(self,
                 dim_jet: int,
                 dim_lepton: int,
                 dim_met: int,
                 dim_output: int,
                 dim_ffnn: int = 256,
                 num_blocks: int = 1,
                 num_heads: int = 2,
                 depth: int = 32,
                 dropout_rate: float = 0.1,
    ) -> None:
        """
        """
        super().__init__()
        self.dim_jet = dim_jet
        self.dim_lepton = dim_lepton
        self.dim_met = dim_met
        self.dim_ffnn = dim_ffnn
        self.num_blocks = num_blocks
        self.depth = depth
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.dim_output = dim_output

        self.dim_model = num_heads * depth

        # input projectuon

        ## jet projection
        self.jet_projection = ObjWise(
            nn.Linear(dim_jet, dim_ffnn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.LeakyReLU())

        ## lepton projection
        self.lepton_projection = nn.Sequential(
            nn.Linear(dim_lepton, dim_ffnn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.LeakyReLU())

        ## met projection
        self.met_projection = nn.Sequential(
            nn.Linear(dim_met, dim_ffnn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.LeakyReLU())

        ## merger
        self.merger = ParticleFlowMerger()

        encoder_attention_blocks = []
        for _ in range(num_blocks):
            block = SelfAttentionBlock(self.dim_model, num_heads, dim_ffnn,
                                       dropout_rate)
            encoder_attention_blocks.append(block)
        self.encoder_attention_blocks = nn.ModuleList(encoder_attention_blocks)

        decoder_attention_blocks = []
        for _ in range(num_blocks):
            block = DecoderBlock(self.dim_model, num_heads, dim_ffnn,
                                 dropout_rate)
            decoder_attention_blocks.append(block)
        self.decoder_attention_blocks = nn.ModuleList(decoder_attention_blocks)

        self.output_projection = ObjWise(
            nn.Linear(self.dim_model, dim_ffnn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, dim_output, bias=True))

    def forward(self, batch: TTbarDileptonBatch) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:
            mask:
        Returns:
        """
        # input projection
        jet = self.jet_projection(input=batch.jet,
                                  data_mask=batch.jet_data_mask)

        lepton = self.lepton_projection(batch.lepton)

        met = self.met_projection(batch.met)

        x, lengths, data_mask = self.merger(
            jet=jet,
            jet_lengths=batch.jet_lengths,
            jet_data_mask=batch.jet_data_mask,
            lepton=lepton,
            met=met)

        data_mask = data_mask.to(x.device)

        #
        attention_list0 = []
        for block in self.encoder_attention_blocks:
            x, attention = block(x, data_mask)
            attention_list0.append(attention)

        source = x
        x = jet
        # attention_list1 = []
        # attention_list2 = []
        for block in self.decoder_attention_blocks:
            x, attention1, attention2 = block(source, x, data_mask, batch.jet_data_mask)
            # attention_list1.append(attention1)
            # attention_list2.append(attention2)

        x = self.output_projection(x, batch.jet_data_mask)
        # attention_list1 = torch.stack(attention_list1, dim=1)
        # attention_list2 = torch.stack(attention_list2, dim=1)

        return x  # , (attention_list0, attention_list1, attention_list2)

class ConSAJA(nn.Module):
    def __init__(self,
                 dim_track: int,
                 dim_tower: int,
                 dim_output: int,
                 dim_ffnn: int = 128,
                 num_blocks: int = 6,
                 num_heads: int = 10,
                 depth: int = 32,
                 dropout_rate: float = 0.1,
    ) -> None:
        """
        """
        super().__init__()
        self.dim_track = dim_track
        self.dim_tower = dim_tower
        self.dim_output = dim_output
        self.dim_ffnn = dim_ffnn
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.depth = depth
        self.dropout_rate = dropout_rate

        self.dim_model = num_heads * depth
        # input projectuon

        ## track projection
        self.track_projection = ObjWise(
            nn.Linear(dim_track, dim_ffnn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.LeakyReLU())

        ## tower projection
        self.tower_projection = ObjWise(
            nn.Linear(dim_tower, dim_ffnn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.LeakyReLU())

        ## merger
        self.merger = ConParticleFlowMerger()

        # aggregate
        self.aggregation = ScatterMean()

        attention_blocks = []
        for _ in range(num_blocks):
            block = SelfAttentionBlock(self.dim_model, num_heads, dim_ffnn, dropout_rate)
            attention_blocks.append(block)
        self.attention_blocks = nn.ModuleList(attention_blocks)

        self.output_projection = nn.Sequential(
                nn.Linear(self.dim_model, dim_ffnn, bias=True),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(dim_ffnn, dim_output, bias=True))

    def forward(self, batch: ConBatch) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:
            mask:
        Returns:
        """
        # input projection
        track = self.track_projection(input=batch.track,
                                      data_mask=batch.track_data_mask)

        tower = self.tower_projection(input=batch.tower,
                                      data_mask=batch.tower_data_mask)

        x, lengths, data_mask = self.merger(
                track=track,
                track_lengths=batch.track_lengths,
                track_data_mask=batch.track_data_mask,
                tower=tower,
                tower_lengths=batch.tower_lengths,
                tower_data_mask=batch.tower_data_mask,
                )

        data_mask = data_mask.to(x.device)

        for block in self.attention_blocks:
            x, attention = block(x, data_mask)

        x = self.aggregation(
                x,
                data_mask,
                lengths,
                )

        # classification_head = self.output_projection(x)
        return x #, classification_head


class TTDileptonWithConSAJA(nn.Module):
    def __init__(self,
                 dim_track: int,
                 dim_tower: int,
                 dim_lepton: int,
                 dim_met: int,
                 dim_output: int,
                 dim_ffnn: int = 128,
                 num_blocks: int = 6,
                 num_heads: int = 10,
                 depth: int = 32,
                 dropout_rate: float = 0.1,
                 # pretrained_model: str = '/home/jheo/vts_saja/vts_bs/track/'
                 # 'model_phi2_256_2_2_32_128_0.0003/model_416.tar',
                 pretrained_model: str = None,
                 pretrained_model_freeze: bool = True
) -> None:
        super().__init__()

        self.dim_track = dim_track
        self.dim_tower = dim_tower
        self.dim_lepton = dim_lepton
        self.dim_met = dim_met
        self.dim_output = dim_output
        self.dim_ffnn = dim_ffnn
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.depth = depth
        self.dropout_rate = dropout_rate

        self.dim_model = num_heads * depth

        self.jet_constituents_encoder = ConSAJA(
                dim_track=dim_track,
                dim_tower=dim_tower,
                dim_output=3,
                dim_ffnn=256,
                num_blocks=2,
                num_heads=2,
                depth=32
        )

        if pretrained_model is not None:
            try:
                self.jet_constituents_encoder.load_state_dict(
                    torch.load(pretrained_model,
                               map_location=torch.device('cpu'))['model']
                )
            except Exception as err:
                print(err)

        if pretrained_model_freeze:
            for param in self.jet_constituents_encoder.parameters():
                param.requires_grad = False

        self.jet_projection = ObjWise(
            nn.Linear(64, dim_ffnn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.LeakyReLU())

        ## lepton projection
        self.lepton_projection = nn.Sequential(
            nn.Linear(dim_lepton, dim_ffnn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.LeakyReLU())

        ## met projection
        self.met_projection = nn.Sequential(
            nn.Linear(dim_met, dim_ffnn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, self.dim_model, bias=True),
            nn.LeakyReLU())

        ## merger
        self.merger = ParticleFlowMerger()

        encoder_attention_blocks = []
        for _ in range(num_blocks):
            block = SelfAttentionBlock(self.dim_model, num_heads, dim_ffnn,
                                       dropout_rate)
            encoder_attention_blocks.append(block)
        self.encoder_attention_blocks = nn.ModuleList(encoder_attention_blocks)

        decoder_attention_blocks = []
        for _ in range(num_blocks):
            block = DecoderBlock(self.dim_model, num_heads, dim_ffnn,
                                 dropout_rate)
            decoder_attention_blocks.append(block)
        self.decoder_attention_blocks = nn.ModuleList(decoder_attention_blocks)

        self.output_projection = ObjWise(
            nn.Linear(self.dim_model, dim_ffnn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffnn, dim_output, bias=True))

    def forward(self, batch: (TTDileptonWithConBatch, ConBatch)) -> Tensor:
        event_batch = batch[0]
        jet_batch = batch[1]

        jet = self.jet_constituents_encoder(jet_batch)

        # Merging jets in each event
        seq_idxs = event_batch.batch_idx
        batch_size = len(event_batch.jet_lengths)
        jet = pad_sequence(
            [jet[seq_idxs == idx] for idx in range(batch_size)],
            batch_first=True)

        jet = self.jet_projection(input=jet,
                                  data_mask=event_batch.jet_data_mask)

        lepton = self.lepton_projection(event_batch.lepton)

        met = self.met_projection(event_batch.met)

        x, lengths, data_mask = self.merger(
            jet=jet,
            jet_lengths=event_batch.jet_lengths,
            jet_data_mask=event_batch.jet_data_mask,
            lepton=lepton,
            met=met)

        data_mask = data_mask.to(x.device)

        attention_list0 = []
        for block in self.encoder_attention_blocks:
            x, attention = block(x, data_mask)
            attention_list0.append(attention)

        source = x
        x = jet
        attention_list1 = []
        attention_list2 = []
        for block in self.decoder_attention_blocks:
            x, attention1, attention2 = block(source,
                                              x,
                                              data_mask,
                                              event_batch.jet_data_mask
                                              )
            attention_list1.append(attention1)
            attention_list2.append(attention2)

        x = self.output_projection(x, event_batch.jet_data_mask)
        attention_list1 = torch.stack(attention_list1, dim=1)
        attention_list2 = torch.stack(attention_list2, dim=1)

        return x
