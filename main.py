import time
start_time = time.time()

import torch
import os, shutil
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import wandb
import warnings
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
wandb.init(project='xyz', entity='abc')

from utils import read_config, delete_coeff, delete_descriptor, delete_overlap, \
  create_coeff, create_descriptor, create_overlap, create_dataset, evaluation
from models import GCN, NN_Simple
from train import train_model_GCN, train_model_NN

device = torch.device("cuda")
print(f"Using {device}")

# read config
config_path = "./gnn_dft_config.json"
config = read_config(config_path)

# model
embedding_size = config['model']['embedding_size']
num_layers = config['model']['number_of_layers']
edge_hidden_size = config['model']['edge_hidden_size']
model_type = config['model']['model_type']
# training
learning_rate = config['training']['learning_rate']
train_batch_size = config['training']['train_batch_size']
loss_factor = config['training']['loss_factor']
vxc_loss_factor = loss_factor[0]
exc_loss_factor = loss_factor[1]
max_train_epoch = config['training']['max_train_epoch']
kfold_cv = config['training']['KFold_CV']
repeat_train_init = config['training']['repeat_train']
chkpt_int = config['training']['checkpt_interval']
restart = config['training']['restart']
restart_fname = config['training']['restart_chkfile']
# data
root = config['data']['dataset_root_folder']
geom_folder = config['data']['dataset_folder']
dataset_file = config['data']['dataset_file']
train_ratio = config['data']['train_ratio']
val_ratio = config['data']['val_ratio']
nlm = config['data']['nlm']  
create_nlm = config['data']['create_nlm']
datatype = config['data']['type']
# experimenmt
log_dir = config['experiment']['log_dir']
seed = config['experiment']['seed']

# set seeds
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
generator = torch.Generator().manual_seed(seed)

# create log directory
log_dir = log_dir + f"/model_{model_type}_lr_{learning_rate}_{embedding_size}_{num_layers}_{edge_hidden_size}_{max_train_epoch}_repeat_{repeat_train_init}_batch_{train_batch_size}_lf_{vxc_loss_factor}"
if not os.path.exists(log_dir):
  os.makedirs(log_dir)
  print(f"Directory '{log_dir}' created.")
shutil.copy(config_path, log_dir + "/config.json")

if create_nlm:
  delete_coeff(dataset_file,geom_folder)
  delete_overlap(dataset_file,geom_folder)
  delete_descriptor(dataset_file,geom_folder,descriptor_file='descriptors.csv')
  delete_descriptor(dataset_file,geom_folder,descriptor_file='c_nlm.csv')
  create_overlap(device,nlm=nlm,dataset_file=dataset_file,geom_folder=geom_folder)
  create_coeff(device,nlm=nlm,dataset_file=dataset_file,geom_folder=geom_folder)
  create_descriptor(device,nlm=nlm,dataset_file=dataset_file,geom_folder=geom_folder)

dataset = create_dataset(root=root,dataset_file=dataset_file, geom_folder=geom_folder, type=datatype)

if datatype == "v1":
  num_features = dataset.num_features
elif datatype == "v2":
  num_features = dataset[0][0].shape[0]

# Split the dataset into train, validation, and test sets
data_size = len(dataset)
train_size = int(data_size * train_ratio)
val_size = int(data_size * val_ratio)
test_size = data_size - train_size - val_size

NUM_GRAPHS_PER_TRAIN_BATCH = train_batch_size
NUM_GRAPHS_PER_TEST_BATCH = 1

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator)
# Print sizes of each dataset split
print(f"Train set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Wrap data in a data loader
train_loader = DataLoader(train_dataset,batch_size=NUM_GRAPHS_PER_TRAIN_BATCH, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=NUM_GRAPHS_PER_TEST_BATCH, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=NUM_GRAPHS_PER_TEST_BATCH, shuffle=False)
all_loader = DataLoader(dataset,batch_size=NUM_GRAPHS_PER_TRAIN_BATCH,shuffle=True)

def save_ckp(state, checkpoint_dir):
  f_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
  torch.save(state, f_path)
  return

def load_ckp(checkpoint_fpath, model, optimizer):
  checkpoint = torch.load(checkpoint_fpath)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  start_epoch  = checkpoint['epoch']
  return model, optimizer, start_epoch


if model_type == "GCN":
  model = GCN(num_features, embedding_size=embedding_size, num_layers=num_layers)
elif model_type == "NNSimple":
  model = NN_Simple(num_features, embedding_size=embedding_size, num_layers=num_layers)
else:
   print("Currently only uses GCN and NNsimple")

print(model)
print(f"number of model parameters : {sum(p.numel() for p in model.parameters())}")
print(f"number of trainable model parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Define loss function and optimizer
loss_fn = torch.nn.MSELoss()
print(f"Loss function is {loss_fn}")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, betas=(0.2,0.222))
print(f"optimizer is {optimizer}")

vxc_loss_factor = torch.tensor(vxc_loss_factor, dtype=torch.float)
exc_loss_factor = torch.tensor(exc_loss_factor, dtype=torch.float)

print(f"Vxc loss factor : {vxc_loss_factor}")
print(f"Exc loss factor : {exc_loss_factor}")

model = model.to(device)

MAX_TRAINING_EPOCH = max_train_epoch
fig_dir = log_dir + "/figs"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
    print(f"Directory '{fig_dir}' created.")

log = {k: [] for k in ['epoch', 'train_exc_loss', 'val_exc_loss',
                       'train_vxc_loss', 'val_vxc_loss',
                       'train_loss', 'val_loss']}
print("-----------------------------------------------------------------------")

print(f"Starting vxc training with factor: {vxc_loss_factor}")
start_epoch = 0

if restart:
  if restart_fname:
    ckp_path = restart_fname
  else:
    ckp_path = os.path.join(log_dir,'checkpoint.pt')
  print(f"Restarting from {ckp_path}")
  model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)
LOSSES = []

for epoch in range(start_epoch,MAX_TRAINING_EPOCH):
  log['epoch'].append(epoch)
  if model_type=="GCN":
    train_exc_loss, train_vxc_loss, train_loss = train_model_GCN(train_loader, model, optimizer, loss_fn, exc_loss_factor, vxc_loss_factor, device, mode=True)
  elif model_type=="NNSimple":
    train_exc_loss, train_vxc_loss, train_loss = train_model_NN(train_loader, model, optimizer, loss_fn, exc_loss_factor, vxc_loss_factor, device, mode=True)

  log['train_exc_loss'].append(train_exc_loss)
  log['train_vxc_loss'].append(train_vxc_loss)
  log['train_loss'].append(train_loss)
      
  if model_type=="GCN":
    val_exc_loss, val_vxc_loss, val_loss = train_model_GCN(val_loader, model, optimizer, loss_fn, exc_loss_factor, vxc_loss_factor, device, mode=False)
  elif model_type=="NNSimple":
    val_exc_loss, val_vxc_loss, val_loss = train_model_NN(val_loader, model, optimizer, loss_fn, exc_loss_factor, vxc_loss_factor, device, mode=False)

  log['val_exc_loss'].append(val_exc_loss)
  log['val_vxc_loss'].append(val_vxc_loss)
  log['val_loss'].append(val_loss)
  
  wandb.log({'Train loss': train_loss, 'Validation Loss': val_loss, 'Train Exc Loss': train_exc_loss, 'Validation Exc Loss': val_exc_loss, 'Train Vxc Loss': train_vxc_loss, 'Validation Vxc Loss': val_vxc_loss})

  if epoch % 100 == 0:
    print(f"Epoch {epoch} | Train Exc Loss: {train_exc_loss:.6f} | Validation Exc Loss: {val_exc_loss:.6f}")
    print(f"Train vxc Loss: {train_vxc_loss:.6f} | Validation vxc Loss: {val_vxc_loss:.6f}")
    print(f"Train combined Loss: {train_loss:.6f} | Validation combined Loss: {val_loss:.6f}")
  if (epoch + 1) % chkpt_int == 0:
    checkpoint = {
      'epoch': epoch + 1,
      'model_state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict()
    }
    save_ckp(checkpoint, log_dir)
    print(f"Checkpoint saved at epoch {epoch}")

  if train_exc_loss < 1e-6:
    print("Early Stopping")
    break

import pandas as pd
log = pd.DataFrame(log)
# save losses log
losses_file_path = log_dir + "/losses_log.csv"
log.to_csv(losses_file_path, index=False)

colors = ["#EC7063","#AF7AC5","#5499C7","#52BE80","#F4D03F","#EB984E","#60151D","#090E76","#AF0505","#FF00FF","#157405"]
plt.figure(figsize=(10, 5))
plt.plot(range(len(log['train_exc_loss'])), log['train_exc_loss'], label="Training Exc Loss", color=colors[2])
plt.plot(range(len(log['val_exc_loss'])), log['val_exc_loss'], label="Validation Exc Loss", color=colors[3])
plt.plot(range(len(log['train_vxc_loss'])), log['train_vxc_loss'], label="Training Vxc Loss", color=colors[4])
plt.plot(range(len(log['val_vxc_loss'])), log['val_vxc_loss'], label="Validation Vxc Loss", color=colors[5])
plt.plot(range(len(log['train_loss'])), log['train_loss'], label="Training Combined Loss", color=colors[0])
plt.plot(range(len(log['val_loss'])), log['val_loss'], label="Validation Combined Loss", color=colors[1])
plt.xlabel("Epochs")
plt.yscale("log")
plt.ylabel("Loss")
plt.title("Training and Validation Losses Over Epochs")
plt.legend()
plt.savefig(fig_dir + "/Train_Valid_Loss_Fig.png", dpi=500)
plt.show()

torch.save(model.state_dict(), log_dir + "/model_state_dict_save")

evaluation(model,model_type,test_loader,device,fig_dir + "/eval_test_exc.png")
evaluation(model,model_type,all_loader,device, fig_dir + "/eval_all_exc.png")

end_time = time.time()
print("Execution time: {:.2f} seconds".format(end_time - start_time))