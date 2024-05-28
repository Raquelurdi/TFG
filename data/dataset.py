from __future__ import print_function
from __future__ import division
import torch
from torch.utils.data import Dataset
import logging
import pytorch_lightning as pl
import numpy as np
from functools import wraps
from time import time
import glob, os, pickle as pkl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class tDataset(Dataset):

    def __init__(self, data, logger=None, transform=None, sequence=True, lime=False, prod=False):
        """
        data es el array "data" que proviene de la funcion load que te copio aqui abajo.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.transform = transform
        self.data = []
        self.ids = []
        self.lime = lime
        for i in range(len(data)):
            self.ids.append(data[i][-1])
            if not prod:
                self.data.append(
                    (data[i][:-2], data[i][-2]) # ([array de caracts.], clase)
                )
            else:
                self.data.append(
                    (data[i][:-2], -1) # ([array de caracts.], clase)
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        data = [0,1,2,3,4,5,6]
        bs = 2

        2,6

        get(2)
        get(6)

        """
        data_row = self.data[idx]
        info, labels = data_row
        if self.lime:
            info = torch.tensor(info, dtype=torch.float)
            # labels = torch.tensor(int(labels))
        else:
            info = torch.tensor(info, dtype=torch.float)
        labels = torch.tensor(int(labels))
        sample = {
            "row": info,
            "label": labels,
            "id": self.ids[idx],
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

class tLSTMDataset(Dataset):

    def __init__(self, data, logger=None, transform=None, sequence=True, lime=False):
        """
        data es el array "data" que proviene de la funcion load que te copio aqui abajo.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.transform = transform
        self.data = []
        self.ids = []
        self.lengths = []
        self.lime = lime
        for i in range(len(data)):
            self.ids.append(data[i][-1])
            self.lengths.append(len(data[i][:-2]))
            self.data.append(
                (data[i][:-2], data[i][-2]) # ([array de caracts.], clase)
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        data = [0,1,2,3,4,5,6]
        bs = 2

        2,6

        get(2)
        get(6)

        """
        data_row = self.data[idx]
        info, labels = data_row
        if self.lime:
            info = torch.tensor(info, dtype=torch.float)
            # labels = torch.tensor(int(labels))
        else:
            info = torch.tensor(info, dtype=torch.float)
        labels = torch.tensor(int(labels))
        sample = {
            "row": info,
            "label": labels,
            "id": self.ids[idx],
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

def load_dict_class(path):
    res = {}
    number_to_class = {}
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.strip()
        legajo, c = line.split(" ")
        c = int(c)
        res[legajo] = c
        number_to_class[c] = legajo
    return res, number_to_class

def load_RWs(opts, prod=False):
    """
    # Expd-ID		  10k feats LABEL
    :param opts:
    :return:
    """
    def load(path):
        print("Loading {}".format(path))
        f = open(path, "r")
        lines = f.readlines()[1:]#[:100]
        f.close()
        data, labels = [], []
        count_mal = 0
        fnames = []

        for line in lines:
            line = line.strip()
            fname = line.split()[0]
            fnames.append(fname)

            feats = line.split()[1:]
            label = int(feats[-1])
            labels.append(label)
            feats = feats[:-1]
            feats = [float(f) for f in feats[:opts.num_feats]]
            f = np.array(feats)
            f = f / f.sum()
            # print(fname, np.mean(feats), f.mean())
            if any(f < 0):
                raise Exception("TFIDF NEGATIVO")
            f = list(f)
            data.append(feats)

        classes = set()
        for i in range(len(labels)):
            data[i].append(labels[i])
            data[i].append(fnames[i])
            classes.add(labels[i])
        print(classes)
        len_feats = len(data[0]) - 2
        return data, len_feats, len(classes)
    
    def loadLSTM(path, class_dict):
        print("Loading {}".format(path))
        files = glob.glob(os.path.join(path, "*idx"))
        data, labels = [], []
        count_mal = 0
        fnames = []

        for pfile in files:
            feats = pkl.load(open(pfile, "rb"))
            fname = pfile.split(("/"))[-1]
            fnames.append(fname)

            label = class_dict[fname.split("_")[-1].split(".")[0]]
            labels.append(label)
            feats = feats[:,:opts.num_feats]
            len_feats = len(feats[0])
            f = np.array(feats)
            f = (f.T / f.sum(axis=1)).T
            if np.any(f < 0):
                raise Exception("TFIDF NEGATIVO")
            f = list(f)
            feats = list(feats)
            data.append(feats)

        classes = set()
        for i in range(len(labels)):
            data[i].append(labels[i])
            data[i].append(fnames[i])
            classes.add(labels[i])
        print(classes)
        # len_feats = len(data[0]) - 2
        return data, len_feats, len(classes)
    
    if prod:
        path_prod = opts.prod_data
        if opts.model == "MLP":
            data_prod, _, _ = load(path_prod)
        else:
            # TODO delete class_dict for prod
            data_prod, _, _ = loadLSTM(path_prod, class_dict)
        return data_prod

    # Class dict
    path_class_dict = opts.class_dict
    class_dict, number_to_class = load_dict_class(path_class_dict)

    path_tr = opts.tr_data
    if opts.model == "MLP":
        data_tr, len_feats, classes = load(path_tr)
    else:
        data_tr, len_feats, classes = loadLSTM(path_tr, class_dict)
    data_te = None
    if not opts.LOO:
        if opts.do_test:
            path_te = opts.te_data
            if opts.model == "MLP":
                data_te, _, _ = load(path_te)
            else:
                data_te, _, _ = loadLSTM(path_te, class_dict)
                # [print(x[-1], x[-2]) for x in data_te]
        # elif opts.do_prod: #TODO te + prod?
        #     path_te = opts.prod_data
        #     if opts.model == "MLP":
        #         data_te, _, _ = load(path_te)
        #     else:
        #         data_te, _, _ = loadLSTM(path_te, class_dict)
    else:
        data_te = None
    
    return data_tr, data_te, len_feats, classes, class_dict, number_to_class

def get_groups(p:str, classes:list):
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    res = []
    for line in lines:
        l, c, ini, fin = line.strip().split(" ")
        if c not in classes:
            continue
        ini, fin = int(ini), int(fin)
        res.append([l, c, ini, fin])
    return res

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[ %r] took: %2.4f sec' % \
          (f.__name__, kw, te-ts))
        return result
    return wrap

# @timing
def search_group(groups:list, npage:int, l:str):
    for lgroup, c, ini, fin in groups:
        if ini <= npage <= fin:
            if l == lgroup:
                return ini, fin
    raise Exception(f'Group for {npage} not found')

# @timing
def search_pages_tfidf(data:list, ini, fin, legajo:str):
    train = []
    for i in data:
        npage = int(i[-1].split("_")[2])
        l_page = i[-1].split("_")[0]
        if ini <= npage <= fin or legajo != l_page:
            continue
        train.append(i)
    return train


class TextDataset(pl.LightningDataModule):

    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None, opts=None, n_test=None, info=None, legajo=""):
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
        # self.setup(opts)
        self.opts = opts
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        
        if opts.LOO:
            self.data_tr_dev, self.data_test, self.len_feats, self.num_classes, self.class_dict, self.number_to_class = info
            self.data_test = [self.data_tr_dev[n_test]]
            if opts.path_file_groups != "":
                groups = get_groups(opts.path_file_groups, opts.classes)
                page_test = int(self.data_test[0][-1].split("_")[2])
                ini, fin = search_group(groups, page_test, legajo)
                self.data_tr_dev = search_pages_tfidf(self.data_tr_dev, ini, fin, legajo)
                print(f'Data: {len(self.data_tr_dev)} samples')
            else:
                self.data_tr_dev = self.data_tr_dev[:n_test] + self.data_tr_dev[n_test+1:]
        else:
            self.data_tr_dev, self.data_test, self.len_feats, self.num_classes, self.class_dict, self.number_to_class = load_RWs(self.opts)
            if self.opts.do_prod:
                self.data_prod = load_RWs(self.opts, prod=True)

        # print(self.data_tr_dev)
    def setup(self, stage):
        print("-----------------------------------------------")
        if self.opts.model == "MLP":
            self.cancerDt_train = tDataset(self.data_tr_dev, transform=self.train_transforms)
            self.cancerDt_val = tDataset(self.data_tr_dev, transform=self.val_transforms)
            if self.opts.do_test or self.opts.LOO:
                self.cancerDt_test = tDataset(self.data_test, transform=None)
            if self.opts.do_prod:
                self.cancerDt_prod = tDataset(self.data_prod, transform=None, prod=True)
        else:
            self.cancerDt_train = tLSTMDataset(self.data_tr_dev, transform=self.train_transforms)
            self.cancerDt_val = tLSTMDataset(self.data_tr_dev, transform=self.val_transforms)
            if self.opts.do_test or self.opts.LOO:
                self.cancerDt_test = tLSTMDataset(self.data_test, transform=None)
            if self.opts.do_prod:
                pass # TODO prod for LSTM models

    def train_dataloader(self):
        if self.opts.model == "MLP":
            trainloader_train = torch.utils.data.DataLoader(self.cancerDt_train, batch_size=self.opts.batch_size, shuffle=True, num_workers=0)
        else:
            trainloader_train = torch.utils.data.DataLoader(self.cancerDt_train, batch_size=self.opts.batch_size, shuffle=True, num_workers=0, collate_fn=PadSequence())
            
        return trainloader_train
    
    def val_dataloader(self):
        if self.opts.model == "MLP":
            trainloader_train = torch.utils.data.DataLoader(self.cancerDt_val, batch_size=self.opts.batch_size, shuffle=False, num_workers=0)
        else:
            trainloader_train = torch.utils.data.DataLoader(self.cancerDt_val, batch_size=self.opts.batch_size, shuffle=False, num_workers=0, collate_fn=PadSequence())
        return trainloader_train
    
    def test_dataloader(self):
        if self.opts.model == "MLP":
            trainloader_train = torch.utils.data.DataLoader(self.cancerDt_test, batch_size=self.opts.batch_size, shuffle=False, num_workers=0)
        else:
            trainloader_train = torch.utils.data.DataLoader(self.cancerDt_test, batch_size=self.opts.batch_size, shuffle=False, num_workers=0, collate_fn=PadSequence())
        return trainloader_train
    
    def predict_dataloader(self):
        if self.opts.model == "MLP":
            trainloader_train = torch.utils.data.DataLoader(self.cancerDt_prod, batch_size=self.opts.batch_size, shuffle=False, num_workers=0)
        else:
            trainloader_train = torch.utils.data.DataLoader(self.cancerDt_prod, batch_size=self.opts.batch_size, shuffle=False, num_workers=0, collate_fn=PadSequence())
        return trainloader_train

class PadSequence:
    def __call__(self, batch):
		# batch : "row": info,
        # "label": labels,
        # "id": self.ids[idx],
        # print(batch)
        # row = batch['row']
        # label = batch['label']
        # id = batch['id']
		# Get each sequence and pad it
        sequences = [x['row'] for x in batch]
        labels = [x['label'] for x in batch]
        ids = [x['id'] for x in batch]
        ls = zip(sequences, labels, ids)
        ls = sorted(ls, key=lambda x: x[0].shape[0], reverse=True)
        sorted_batch, labels, ids = [], [], []
        for s, l, i in ls:
            sorted_batch.append(s)
            labels.append(l)
            ids.append(i)
        # print(sorted_batch)
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sorted_batch, batch_first=True)
        lengths = torch.LongTensor([len(x) for x in sorted_batch])
        # print(lengths)
        # print(len(sorted_batch))
        # sorted_batch = torch.stack(sorted_batch)
        # print(sorted_batch.shape)
        packed_input = pack_padded_sequence(sequences_padded, lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        # Also need to store the length of each sequence
		# This is later needed in order to unpad the sequences
		# Don't forget to grab the labels of the *sorted* batch
        labels = torch.LongTensor(labels)
        # labels = label
        # print(sequences_padded, lengths, labels)
        # exit()
        # print(lengths, labels, ids)
        return packed_input, lengths, labels, ids