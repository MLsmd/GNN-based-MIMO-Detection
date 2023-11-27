import torch
import torch.nn as nn
import pandas as pd
import os

from datetime import datetime
import time

from parsers import parsersers_,constellation
from Train_Eval_funcs import train,evaluate
from Data_loader import Data_loader
#from GNN_model import GNN
from GNN_luca import GNN_fully_connected


# initialization

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)
        
args            = vars(parsersers_())
Nr              = args['Nr']
Nt              = args['Nt']
with_mmse       = args['MMSE']
snr_db_min      = args['SNR_dB_min_train']
snr_db_max      = args['SNR_dB_max_train']
num_classes     = args['num_classes']
num_epochs      = args['n_epochs']
iter_gnn        = args['iter_GNN']
num_neuron      = args['num_neuron']
num_su          = args['num_feature_su']
dropout         = args['Dropout']
total_samples   = args['samples']
batch_size      = args['batch_size']

#model = GNN(iter_gnn,num_neuron,num_su,num_classes,Nt*2,with_mmse,dropout).to(device)
model = GNN_fully_connected(n=Nt, m=Nr, in_size=3, out_size=num_classes,
                            size_vn_values = 8, # N_u in [KOM+22]
                            size_edge_embed = 2, # (h_k^T h_j, sigma2_noise) for each edge f_{j,k}
                            n_hidden_layers_prop = 2,
                            size_per_hidden_layer_prop = [num_neuron, num_neuron//2], # l and l/2 in [SML+20]
                            size_edge_values = 8,
                            size_gru_hidden_state=num_neuron, # l in [SML+20]
                            size_agg_embed = 0, # no aggregation embedding here
                            size_out_layer_agg = 8, # N_u
                            n_hidden_layers_readout = 2,
                            size_per_hidden_layer_readout = [num_neuron, num_neuron//2], # N_h1 and N_h2
                            device = device,
                            )

print(f'Nr={Nr},Nt={Nt},snr_db_min={snr_db_min},snr_db_max={snr_db_max},num_epochs={num_epochs},num_neuron={num_neuron},iter_GNN={iter_gnn}')


dataLoader = Data_loader (Nt,Nr,total_samples,batch_size, snr_db_min, snr_db_max,constellation)
train_dataloader = dataLoader.getTrainData()
val_dataloader = dataLoader.getValData()
    
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)

model_name=model.__class__.__name__ 

dt_string = datetime.now().strftime("%d_%H:%M:%S")

# Create directories to save trained models and training reports
for k in ['models','reports']:
    if not os.path.exists(k):
        os.makedirs(k)
    else:
        if not os.path.exists(f'{k}/{model_name}_MMSE_{with_mmse}_{Nr*2}X{Nt*2}_SNR_{snr_db_min}_{snr_db_max}_dB'):
            os.makedirs(f'{k}/{model_name}_MMSE_{with_mmse}_{Nr*2}X{Nt*2}_SNR_{snr_db_min}_{snr_db_max}_dB')
    
f = open(f'reports/{model_name}_MMSE_{with_mmse}_{Nr*2}X{Nt*2}_SNR_{snr_db_min}_{snr_db_max}_dB/log.txt',"w")
f.write(dt_string + "\n")
f.write(str(device) + "\n")
f.close()
report={'epoch':[],'train_SER':[],'val_SER':[]}
best_epoch = 0
val_acc_prev = 0

# Start training epoch
for epoch in range(args['n_epochs']):
    t = time.time()
    loss,train_acc,train_SER,avg_SNR=train(model, device, train_dataloader, optimizer, epoch, criterion, Nt*2, dtype,num_neuron,Nr,constellation)
    val_acc,val_SER=evaluate(model,device, val_dataloader,Nt*2,dtype,num_neuron,constellation)
    
    print(f'epoch{epoch}: loss={loss:.8f},train_acc={train_acc:.8f}, train_SER={train_SER:.8f}, avg_training_SNR={avg_SNR:.4f}')
    print(f'val_acc={val_acc:.8f}, val_SER={val_SER:.8f}, avg_training_SNR={avg_SNR:.4f} \n')
    model.to('cpu')


    # Save best model over epochs
    if (val_acc > val_acc_prev):
        torch.save(model.state_dict(), f'models/{model_name}_MMSE_{with_mmse}_{Nr*2}X{Nt*2}_SNR_{snr_db_min}_{snr_db_max}_dB/model.pkl')
        val_acc_prev = val_acc
        train_acc_prev = train_acc
        best_epoch=epoch
    elif (val_acc == val_acc_prev):
        if train_acc >= train_acc_prev:
            torch.save(model.state_dict(), f'models/{model_name}_MMSE_{with_mmse}_{Nr*2}X{Nt*2}_SNR_{snr_db_min}_{snr_db_max}_dB/model.pkl')
            train_acc_prev = train_acc
            best_epoch=epoch
    
    model.to(device)
    
    f = open(f'reports/{model_name}_MMSE_{with_mmse}_{Nr*2}X{Nt*2}_SNR_{snr_db_min}_{snr_db_max}_dB/best_epoch.txt', "w")
    f.write(str(best_epoch))
    f.close()

    report['epoch'].append(epoch)
    report['train_SER'].append(train_SER)
    report['val_SER'].append(val_SER)
    
    df_report=pd.DataFrame(report)
    df_report.set_index('epoch',inplace=True)
    df_report.to_csv(f'reports/{model_name}_MMSE_{with_mmse}_{Nr*2}X{Nt*2}_SNR_{snr_db_min}_{snr_db_max}_dB/report.csv',index=True)
    
    
    elapsed = time.time() - t
    f = open(f'reports/{model_name}_MMSE_{with_mmse}_{Nr*2}X{Nt*2}_SNR_{snr_db_min}_{snr_db_max}_dB/log.txt',"a")
    f.write("time per epoch: " + str(elapsed) + "\n")
    f.close()
    