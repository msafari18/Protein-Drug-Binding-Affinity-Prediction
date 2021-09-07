import torch
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model_class import MyModel


class data(Dataset):
    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def __getitem__(self, index):
        data = np.array(self.data.loc[index])
        sample = data
        return sample

    def __len__(self):
        self.len = self.data.shape
        return len(self.data)

def pre_process_SMILE(final_SMILES):
    char_num = {'7': 1, '3': 2, '=': 3, '9': 4, 'O': 5, 'B': 6, 'I': 7, 'N': 8, 'H': 9, '(': 10, ')': 11, '2': 12,
                'F': 13, 'C': 14, '5': 15, '1': 16, '/': 17, '@': 18, 'l': 19, '8': 20, '[': 21, '4': 22, 'r': 23,
                ']': 24, 'P': 25, 'S': 26, '6': 27, '#': 28, '\\': 29, '.': 30}
    SMILES_length = 95
    pre_procesed_SMILES = []
    for i in final_SMILES:
        x = []
        for j in i:
            x.append(char_num[j])
        if len(x) > SMILES_length:
            x = x[:SMILES_length]
        elif len(x) < SMILES_length:
            for j in range(SMILES_length - len(x)):
                x.append(0)
        pre_procesed_SMILES.append(x)
    return pre_procesed_SMILES

def pre_process_proteins(final_proteins):
    char_num = {}
    proteins_length = 450
    word_length = 3

    unique_word = {}
    for protein in final_proteins:
        for n in range(0, (len(protein) - word_length + 1), word_length):
            if protein[n:n + word_length] not in unique_word:
                unique_word[protein[n:n + word_length]] = 1
            else:
                unique_word[protein[n:n + word_length]] += 1

    unique_word = {k: v for k, v in sorted(unique_word.items(), key=lambda item: item[1])}
    word_num = {}
    for n, i in enumerate(list(unique_word)):
        word_num[i] = n + 1

    pre_procesed_proteins = []
    length = []
    for protein in final_proteins:
        x = []
        for n in range(0, (len(protein) - word_length + 1), word_length):
            x.append(word_num[protein[n:n + word_length]])
        length.append(len(x))
        if len(x) > proteins_length:
            x = x[:proteins_length]
        elif len(x) < proteins_length:
            for j in range(proteins_length - len(x)):
                x.append(0)
        pre_procesed_proteins.append(x)

    return pre_procesed_proteins



if __name__ == "__main__":
    x = sys.argv
    path1 = x[1]
    path2 = x[2]

    with open(path1) as f:
        lines = [line.rstrip() for line in f]

    proteins = [i.split('\t')[0] for i in lines ]
    drugs = [i.split('\t')[1] for i in lines]
    pre_procesed_proteins = pre_process_proteins(proteins)
    pre_procesed_SMILES = pre_process_SMILE(drugs)
    batch_size = 1
    test_set_p = data(pre_procesed_proteins)
    test_loader_p = DataLoader(dataset=test_set_p,
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=True)

    test_set_s = data(pre_procesed_SMILES)
    test_loader_s = DataLoader(dataset=test_set_s,
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=True)

    final_model = MyModel()
    final_model.load_state_dict(torch.load("myModel_final_d", map_location=torch.device('cpu')))
    device = torch.device('cpu')
    y_pred = []
    with torch.no_grad():
        final_model.eval()
        for datap, datas in zip(test_loader_p, test_loader_s):
            outputs = final_model(datap, datas)
            y_pred += outputs.tolist()
    y_pred = np.array(y_pred)
    y_pred = [1 if i>=7 else 0 for i in y_pred]
    with open(path2, 'w') as f1:
        for i in y_pred:
            f1.write(str(i) + "\n")





