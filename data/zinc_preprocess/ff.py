# -*- coding: utf-8 -*-
import pandas as pd
from rdkit import Chem
from data_structs import filter_mol
import numpy as np

smi_read = 'zinc.txt' 
smiles_list = []
with open(smi_read, 'r') as f:
    for line in f:
        smiles = line.split(" ")[0]
        smiles_list.append(smiles)

print('compounds number:',len(smiles_list))
#删除含有‘.’的分子
def preprocess(data):
    smiles_list =[]
    for smi  in data:
        mol = Chem.MolFromSmiles(smi)
        if filter_mol(mol):
            smile = Chem.MolToSmiles(mol,isomericSmiles=False)
            smiles_list.append(smile)
    mm = int(len(data) - len(smiles_list))
    print(' {} moleculars were deleted from {} '.format(mm,len(data)))
    return smiles_list

preprocess_smi = preprocess(smiles_list)
new_smi = []
for smi in preprocess_smi:
    if smi.find('.')== -1:
        new_smi.append(smi)
print('deleted mixtures compound, remain :',len(new_smi))

#删除异构体的分子
mols = [Chem.MolFromSmiles(c) for c in new_smi]
smis =[]
for mol in mols:
    smi = Chem.MolToSmiles(mol,isomericSmiles=False)
    smis.append(smi)

#删除重复扽子
smis =  pd.DataFrame(smis) 
new_smi = smis.drop_duplicates(keep='last')
print('deleted duplicated stuctures, remain:',len(new_smi))

#删除抑制剂分子
cyp_inhibitor = pd.read_table(r'../cyp_preprocess/finetune_train.txt',header=None)
compoundss = np.array(pd.concat([cyp_inhibitor,new_smi])).tolist()
compounds = []
for item in compoundss:
    for i in item:
        compounds.append(i)
labels = [0]*len(cyp_inhibitor) +  [1]*len(new_smi)
df = pd.DataFrame(zip(compounds, labels), columns=['compound', 'label'])
df = df.drop_duplicates(subset=['compound'], keep='first')
df= df.astype(str)
smi_chembl_2 = df[df['label'].str.contains('1')]
new_smi_2 = smi_chembl_2['compound']
print('deleted  cyp_inhibitor, remain:',len(new_smi_2))

save_path = 'all.txt'
with open(save_path, 'w') as f:
    for smiles in new_smi_2 :
        f.write(smiles + "\n")
