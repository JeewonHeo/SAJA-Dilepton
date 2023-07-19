import os
import torch
import yaml
import uproot
from torch.utils.data import DataLoader
from saja import MinMaxScaler
from saja import TTbarDileptonSAJA
from saja import TTbarDileptonDataset
from saja import object_wise_cross_entropy


def train(model, train_loader, opt, device):
    torch.set_grad_enabled(True)
    model.train()
    train_loss = 0
    for batch in train_loader:
        opt.zero_grad()
        batch = batch.to(device)
        logits = model(batch)
        # Using custom loss
        loss = object_wise_cross_entropy(logits,
                                         batch.target,
                                         torch.logical_not(
                                             batch.jet_data_mask
                                             ),
                                         batch.jet_lengths)
        loss.backward()
        opt.step()
        train_loss += loss.item() * len(batch.target)
    return train_loss

def validation(model, valid_loader, device):
    torch.set_grad_enabled(False)
    model.eval()
    valid_loss = 0
    for batch in valid_loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = object_wise_cross_entropy(logits,
                                         batch.target,
                                         torch.logical_not(
                                             batch.jet_data_mask
                                             ),
                                         batch.jet_lengths,
                                         reduction='none').sum()
        valid_loss += loss.item()
    return valid_loss

if __name__=='__main__':
    with open('example_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.CLoader)

        branches = config['branches']
        hyper_params = config['hyper_params']

    example_rootfile = "example_dilepton.root"

    f = uproot.open(example_rootfile)


    tree_path = 'delphes'
    dataset = TTbarDileptonDataset(example_rootfile,
                                   tree_path,
                                   **config['branches'])

    train_dataset = dataset
    valid_dataset = dataset


    save_path = "model_output/tutorial_model"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    scaler = MinMaxScaler(
            branches['jet_branches'],
            branches['lep_branches'],
            branches['met_branches'],
            )
    scaler.fit(train_dataset)
    torch.save(scaler, f'{save_path}/scaler.pt')

    # scaler = torch.load(f'{save_path}/scaler.pt')  # you must use the scaler that fitted in the training dataset
    scaler.transform(train_dataset)
    scaler.transform(valid_dataset)


    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=hyper_params['batch_size'],
                              collate_fn=train_dataset.collate,
                              **config['loader_args'],
                              )

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=512,
                              collate_fn=valid_dataset.collate,
                              )
    device = torch.device("cuda" if (torch.cuda.is_available()) else 'cpu')


    model = TTbarDileptonSAJA(dim_jet=len(branches['jet_branches']),
                              dim_lepton=len(branches['lep_branches']),
                              dim_met=len(branches['met_branches']),
                              dim_output=2,  # 0: other, 1: b-parton matched jet
                              dim_ffnn=hyper_params['dim_ffnn'],
                              num_blocks=hyper_params['num_blocks'],
                              num_heads=hyper_params['num_heads'],
                              depth=hyper_params['depth'],
            ).to(device)  # Send to devcie (GPU or CPU..., whatever you defined)

    opt = torch.optim.Adam(model.parameters(),
                           lr=hyper_params['learning_rate'])

    for epoch in range(config['n_epoch']):
        train_loss = train(model, train_loader, opt, device)
        valid_loss = validation(model, valid_loader, device)
        print(f"Epoch {epoch} Done")
        print(f"  {train_loss = :.4f}\t{valid_loss = :.4f}")
