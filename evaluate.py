# Evaluation Function
from operator import index
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
from config import *

def evaluate(model, test_loader, device, cm_name):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for (text, labels), _ in test_loader:
                labels = labels.type(torch.LongTensor)           
                labels = labels.to(device)
                text = text.type(torch.LongTensor)
                text = text.to(device)
                output = model(text, labels)
                _, output = output
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(labels.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    # save confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['NEED', 'NON-NEED'])
    ax.yaxis.set_ticklabels(['NEED', 'NON-NEED'])
    plt.savefig(des_folder + cm_name)
    return np.array(y_pred, dtype=int)

    