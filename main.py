import torch
from torch import optim
from model import BERT
from data_loader import load_data
from train import train
import matplotlib.pyplot as plt
from utils import load_metrics, load_checkpoint
from evaluate import evaluate
from config import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model = BERT().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
train_iter, valid_iter, test_iter = load_data(device=device)
print("start training...")
train(model=model, optimizer=optimizer, train_loader=train_iter, valid_loader=valid_iter, num_epochs=num_epochs, eval_every=len(train_iter) // 2, file_path=des_folder, device=device, best_valid_loss=float("Inf"))

# save the training iteration figure
train_loss_list, valid_loss_list, global_steps_list = load_metrics(des_folder + '/metrics.pt', device)
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig(des_folder + 'train_iter.png')
plt.cla()

# evaluate
best_model = BERT().to(device)
load_checkpoint(des_folder + '/model.pt', best_model, device)
evaluate(best_model, test_iter, device, des_folder)