import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from model import BERT
from data_loader import load_data
from train import train
from utils import load_metrics, load_checkpoint
from evaluate import evaluate
from config import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model = BERT().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
train_iter, valid_iter, test_iter, novel_test_iter = load_data(device=device)
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

# evaluate the classifier
best_model = BERT().to(device)
load_checkpoint(des_folder + '/model.pt', best_model, device)
_ = evaluate(best_model, test_iter, device, 'eval_classifier.png')

# evaluate the whole process
y_pred = evaluate(best_model, novel_test_iter, device, 'eval_novel.png')
# save the predicted results
data_novel_test = pd.read_csv(data_novel_test_path)
data_novel_test['pred'] = np.array(y_pred, dtype=int)
data_novel_test.to_csv(des_folder + 'need_results.csv')