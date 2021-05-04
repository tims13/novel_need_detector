# Evaluation Function
from operator import index
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd

def evaluate(model, test_loader,device, des_folder, data_path = 'data/'):
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
    plt.savefig(des_folder + 'eval.png')

    # save the predicted results
    data_test = pd.read_csv(data_path + 'test.csv')
    data_test['pred'] = np.array(y_pred, dtype=int)
    '''
    data_test_positive = data_test[data_test['pred'] == 1]
    data_test_positive.to_csv(des_folder + 'need_found.csv')
    '''
    data_test.to_csv(des_folder + 'need_results.csv')

def evaluate_novelty(model, novel_loader,device, data_novel_csv_path = 'sentence/novel.csv', data_result_csv_path = 'record/novel_result.csv'):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (text, labels), _ in novel_loader:

                labels = labels.type(torch.LongTensor)           
                labels = labels.to(device)
                text = text.type(torch.LongTensor)  
                text = text.to(device)
                output = model(text, labels)

                _, output = output
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(labels.tolist())

    data_novel = pd.read_csv(data_novel_csv_path)
    data_novel['pred'] = np.array(y_pred, dtype=int)
    data_novel.to_csv(data_result_csv_path, index=False)
    