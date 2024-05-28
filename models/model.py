import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.autograd import Variable
import torch.optim as optim

import pytorch_lightning as pl
import torchmetrics
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class OnevsAllSigmoidLoss(nn.Module):

    def __init__(self, nclasses) -> None:
        super().__init__()
        self.nclasses = nclasses
        self.min_value = 0.00000000000001
    
    def forward(self, probs , y):
        oh = F.one_hot(y, num_classes=self.nclasses)
        oh_negative = torch.logical_not(oh)
        probs = torch.clamp(probs, min=self.min_value)
        neg_probs = torch.clamp((1 - probs), min = self.min_value)
        l = -torch.log(probs)*oh - (torch.log(neg_probs))*oh_negative
        # if torch.any(torch.isnan(l)):
        #     print(probs)
        #     print(y)
        #     print(oh)
        #     print(l)
        #     exit()
        l = torch.sum(l)
        return l

class Net(pl.LightningModule):
    def __init__(self, len_feats, layers=[512,256], n_classes=3, opts=None):
        super(Net, self).__init__()
        self.opts = opts
        self.layers = layers
        self.num_params = 0
        self.batch_size = opts.batch_size
        self.type_seq = opts.type_seq
        if opts.model == "MLP":
            if opts.type_seq == "MLP":
                if layers == [0]:
                    model = [nn.Linear((len_feats), n_classes)]
                else:
                    if len(layers) == 1:
                        self.hidden = nn.Linear(len_feats, layers[0])
                        self.bn = nn.BatchNorm1d(layers[0])
                        self.linear = nn.Linear(layers[0], n_classes)
                    else:
                        model = [nn.Linear((len_feats), layers[0]),
                                nn.BatchNorm1d(layers[0]),
                                nn.ReLU(True),
                                nn.Dropout(opts.DO)
                                ]
                        for i in range(1, len(layers)):
                            model = model + [nn.Linear(layers[i-1], layers[i]),
                                nn.BatchNorm1d(layers[i]),
                                nn.ReLU(True),
                                nn.Dropout(opts.DO)
                                ]
                        model = model + [
                                nn.Linear(layers[-1], n_classes),
                                # nn.Softmax()
                        ]
                if len(layers) == 1 and layers != [0]:
                    for param in self.parameters():
                        self.num_params += param.numel()
                else:
                    self.model = nn.Sequential(*model)
                    for param in self.model.parameters():
                        self.num_params += param.numel()
            else: #transformer 
                nheads = opts.nheads
                dim_feedforward = opts.dim_feedforward
                self.encoder_layer = nn.TransformerEncoderLayer(d_model=len_feats, nhead=nheads, dim_feedforward=dim_feedforward)
                self.lin = nn.Linear(len_feats, n_classes)
                
            
        elif "LSTM" in opts.model:
            self.model_lstm = nn.LSTM(len_feats, layers[0], len(layers), batch_first=True, dropout=opts.DO, bidirectional=True)
            if "voting" in  opts.model:
                self.model_linear = nn.Linear(layers[0]*2, n_classes)      
            else:
                self.model_linear = nn.Linear(layers[0]*2*2, n_classes)     
            print(self.model_lstm)
            print(self.model_linear)
        
        if opts.openset == "onevsall":
            self.sigmoid = nn.Sigmoid()
            self.criterion = OnevsAllSigmoidLoss(nclasses=n_classes)
        elif opts.loss.lower() == "cross-entropy" or opts.loss.lower() == "cr" or opts.loss.lower() == "crossentropy":
            self.criterion = nn.NLLLoss(reduction="mean")
            # criterion = nn.CrossEntropyLoss()
        else:
            raise Exception("Loss function doesnt exists")

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.lr = self.opts.lr

    def onevsall_loss(self,inp):
        return 

    def forward(self, inp):
        """
        [60, 2000]
        return [60, 3] 
        """
        # print(inp)
        if self.opts.model == "MLP":
            if self.opts.type_seq == "MLP":
                if len(self.layers) == 1 and self.layers != [0]:
                    relu = nn.ReLU()

                    prod_w0 = self.hidden(inp)

                    hidden = relu(self.bn(prod_w0))

                    w_final = self.linear(hidden)
                    if self.opts.openset == "onevsall":
                        probs = self.sigmoid(w_final)
                    else:
                        probs = F.log_softmax(w_final, dim=-1)
                    return probs
                if self.opts.openset == "onevsall":
                    x = self.model(inp)
                    probs = self.sigmoid(x)
                else:
                    probs = F.log_softmax(self.model(inp), dim=-1)
                return probs
            else:
                inp = inp.unsqueeze(0)
                # if self.type_seq == "encoder":
                x = self.encoder_layer(inp)
                # elif self.type_seq == "tencoder":
                #     x = self.seq(inp)
                # elif self.type_seq == "mha":
                #     x,_ = self.seq(inp, inp, inp)
                x = x.squeeze(0)
                x = self.lin(x)
                if self.opts.openset == "onevsall":
                    probs = self.sigmoid(x)
                else:
                    probs = F.log_softmax(x, dim=-1)
                return probs
        elif "LSTM" in self.opts.model:
            x, sizes = inp
            packed_output, (hn, cn) = self.model_lstm(x)
            output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
            if "voting" in self.opts.model:
                y = self.model_linear(output)
                return F.log_softmax(y, dim=-1)
            else:
                out_forward = output[:, -1, :]
                out_reverse = output[:, 0, :]
                output = torch.cat((out_forward, out_reverse), 1)
                y = self.model_linear(output)
                return F.log_softmax(y, dim=-1)

    
    def configure_optimizers(self):
        if self.opts.optim == "ADAM":
            optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(self.opts.adam_beta1, self.opts.adam_beta2))

        elif self.opts.optim == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5*(10**-4))
        elif self.opts.optim == "RMSprop":
            optimizer = optim.RMSprop(self.parameters(), lr=self.lr, momentum=0)
        else:
            raise Exception("Optimizer {} not implemented".format(self.opts.optim))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.opts.steps, gamma=self.opts.gamma_step)
        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx):
        # get the inputs
        if self.opts.model == "MLP":
            x = train_batch['row']#.to(device)
            y_gt = train_batch['label']#.to(device)
        else:
            sequences_padded, lengths, y_gt, ids = train_batch
            x = (sequences_padded, lengths)
            # print("y_gt == ", y_gt.shape)
        # print(x.shape)
        bs1 = False
        if self.opts.model == "MLP" and x.shape[0] == 1:
            # print("TRUE -----------------------"*10)
            bs1 = True
            x = torch.cat([x,x], dim=0)
        outputs = self(x)
        if "voting" in self.opts.model:
            new_gt = []
            for i, y in enumerate(y_gt.cpu().detach().numpy()):
                new_gt.extend([y]*lengths[i])
            y_gt = torch.LongTensor(new_gt)
            new_output = []
            for i, o in enumerate(outputs):
                slice = o[:lengths[i],:]
                new_output.append(slice)
            outputs = torch.cat(new_output)
        if bs1:
            outputs = outputs[0]
            outputs = outputs.expand(1,outputs.shape[0])
        loss = self.criterion(outputs, y_gt)
        self.log('train_loss', loss)
        if not self.opts.openset == "onevsall":
            self.train_acc(torch.exp(outputs), y_gt)
        else:
            self.train_acc(outputs, y_gt)
        self.log('train_acc_step', self.train_acc)
        return {'outputs': outputs, 'loss':loss, 'y_gt':y_gt}
    
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.train_acc)
        outputs = []
        gts = []
        losses = []
        for x in outs:
            o = x['outputs']
            # print(o.shape)
            if not self.opts.openset == "onevsall":
                aux = torch.argmax(torch.exp(o), dim=-1)
            else:
                aux = torch.argmax(o, dim=-1)
            outputs.extend(aux)
            gts.extend(x['y_gt']) 
            l = x['loss']
            losses.append(l)
        l = torch.mean(torch.Tensor(l))
        self.log('train_epoch_loss', l)
        outputs = torch.Tensor(outputs)   
        gts = torch.Tensor(gts)   
        acc = (outputs == gts).sum() / gts.size(0)
        print(f'Accuracy train: {acc}')
    
    def validation_step(self, val_batch, batch_idx):
        if self.opts.model == "MLP":
            x = val_batch['row']#.to(device)
            y_gt = val_batch['label']#.to(device)
        else:
            sequences_padded, lengths, y_gt, ids = val_batch
            x = (sequences_padded, lengths)
            # print("y_gt == ", y_gt.shape)
        outputs = self(x)
        if "voting" in self.opts.model:
            new_gt = []
            for i, y in enumerate(y_gt.cpu().detach().numpy()):
                new_gt.extend([y]*lengths[i])
            y_gt = torch.LongTensor(new_gt)
            new_output = []
            for i, o in enumerate(outputs):
                slice = o[:lengths[i],:]
                new_output.append(slice)
            outputs = torch.cat(new_output)
        loss = self.criterion(outputs, y_gt)
        self.log('val_loss', loss)
        if not self.opts.openset == "onevsall":
            self.val_acc(torch.exp(outputs), y_gt)
        else:
            self.val_acc(outputs, y_gt)
        self.log('val_acc_step', self.val_acc)
        return {'outputs': outputs, 'loss':loss, 'y_gt':y_gt}
    
    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('val_acc_epoch', self.val_acc)
        outputs = []
        gts = []
        losses = []
        for x in outs:
            l = x['loss']
            o = x['outputs']
            # print(o.shape)
            if not self.opts.openset == "onevsall":
                aux = torch.argmax(torch.exp(o), dim=-1)
            else:
                aux = torch.argmax(o, dim=-1)
            outputs.extend(aux)
            gts.extend(x['y_gt']) 
            losses.append(l)
        l = torch.mean(torch.Tensor(l))
        self.log('val_epoch_loss', l)
        outputs = torch.Tensor(outputs)   
        gts = torch.Tensor(gts)   
        acc = (outputs == gts).sum() / gts.size(0)
        print(f'Accuracy val: {acc}')
    
    def test_step(self, train_batch, batch_idx):
        # get the inputs
        if self.opts.model == "MLP":
            x = train_batch['row']#.to(device)
            y_gt = train_batch['label']#.to(device)
        else:
            sequences_padded, lengths, y_gt, ids = train_batch
            x = (sequences_padded, lengths)
        outputs = self(x)
        print(outputs)
        if "voting" in self.opts.model:
            new_gt = []
            for i, y in enumerate(y_gt.cpu().detach().numpy()):
                new_gt.extend([y]*lengths[i])
            y_gt = torch.LongTensor(new_gt)
            new_output = []
            for i, o in enumerate(outputs):
                slice = o[:lengths[i],:]
                new_output.append(slice)
            outputs = torch.cat(new_output)
        if not self.opts.openset == "onevsall":
            outputs = torch.exp(outputs)
        # print(torch.argmax(outputs, dim=1), y_gt)
        acc = (torch.argmax(outputs, dim=1) == y_gt).sum() / y_gt.size(0)
        # print("Acc Test epoch: ", acc, y_gt.shape)
        self.test_acc(outputs, y_gt)
        self.log('test_acc_step', self.test_acc)
        return {'outputs': outputs, 'y_gt':y_gt}
    
    def test_epoch_end(self, outs):
        # log epoch metric
        self.log('test_acc_epoch', self.test_acc)
        outputs = []
        gts = []
        for x in outs:
            o = x['outputs']
            # print(o.shape)
            if not self.opts.openset == "onevsall":
                aux = torch.argmax(torch.exp(o), dim=-1)
            else:
                aux = torch.argmax(o, dim=-1)
            outputs.extend(aux)
            gts.extend(x['y_gt']) 
        
        outputs = torch.Tensor(outputs)   
        gts = torch.Tensor(gts)   
        acc = (outputs == gts).sum() / gts.size(0)
        print(f'Accuracy test: {acc}')
        
        return outs
    
    def predict_step(self, train_batch, batch_idx):
        # get the inputs
        if self.opts.model == "MLP":
            x = train_batch['row']#.to(device)
            y_gt = train_batch['label']#.to(device)
        else:
            sequences_padded, lengths, y_gt, ids = train_batch
            x = (sequences_padded, lengths)
        outputs = self(x)

        if "voting" in self.opts.model:
            new_gt = []
            for i, y in enumerate(y_gt.cpu().detach().numpy()):
                new_gt.extend([y]*lengths[i])
            y_gt = torch.LongTensor(new_gt)
            new_output = []
            for i, o in enumerate(outputs):
                slice = o[:lengths[i],:]
                new_output.append(slice)
            outputs = torch.cat(new_output)

            last_ids, last_ys = ids, y_gt
            # Expand
            ids, ys = [], []
            for i in range(len(last_ids)):

                id_i = last_ids[i]
                length_i = lengths[i]
                y_i = last_ys[i]
                for j in range(length_i):
                    #Ej: JMBD4950_page_396_other.idx
                    l, _, range_p, c = id_i.split("_")
                    pini = int(range_p.split("-")[0]) + j
                    c = c.split(".")[0]
                    ys.append(y_i)
                    ids.append(f'{l}_page_{pini}_{c}.idx')
            # print(ids)
            # exit()
            # print(outputs.shape)
            # print(len(ys))
            # ys_tensor = torch.Tensor(ys)
            argmax_outputs = torch.argmax(outputs, dim=1)
            acc = (argmax_outputs == y_gt).sum() / y_gt.size(0)
            ys = y_gt
        #     print(ids)
        #     print(y_gt)
        #     print("Acc perpage predict epoch: ", acc, y_gt.shape)
        argmax_outputs = torch.argmax(outputs, dim=1)
        # print("outputs", argmax_outputs, argmax_outputs.shape)
        # print("y_gt", y_gt)
        # print("ys", ys)
        acc = (torch.argmax(outputs, dim=1) == y_gt).sum() / y_gt.size(0)
        # print("Acc predict epoch: ", acc, y_gt.shape)
        # print(y_gt[0])
        if not self.opts.openset == "onevsall":
            outputs = torch.exp(outputs)
        if "voting" in self.opts.model:
            return {'outputs': outputs, 'y_gt':ys, 'ids':ids} # outputs, ids, ys
        else:
            return outputs


class NetLIME(nn.Module):
    def __init__(self, len_feats, layers=[512,256], n_classes=3, dropout=0, opts=None):
        super(NetLIME   , self).__init__()
        self.layers = layers
        if layers == [0]:
            model = [nn.Linear((len_feats), n_classes)]
        else:
            if len(layers) == 1:
                self.hidden = nn.Linear(len_feats, layers[0])
                self.bn = nn.BatchNorm1d(layers[0])
                self.linear = nn.Linear(layers[0], n_classes)
            else:
                model = [nn.Linear((len_feats), layers[0]),
                         nn.BatchNorm1d(layers[0]),
                         nn.ReLU(True),
                         # nn.Dropout(dropout)
                         ]
                for i in range(1, len(layers)):
                    model = model + [nn.Linear(layers[i-1], layers[i]),
                         nn.BatchNorm1d(layers[i]),
                         nn.ReLU(True),
                         # nn.Dropout(dropout)
                         ]
                model = model + [
                        nn.Linear(layers[-1], n_classes),
                        # nn.Softmax()
                ]
        self.num_params = 0

        if len(layers) == 1 and layers != [0]:
            for param in self.parameters():
                self.num_params += param.numel()
        else:
            self.model = nn.Sequential(*model)
            for param in self.model.parameters():
                self.num_params += param.numel()


    def forward(self, inp):
        """
        [60, 2000]
        return [60, 3] 
        """
        lime = False
        if type(inp) != torch.Tensor:
            inp = torch.tensor(inp, dtype=torch.float32).cuda()
            # print("Entered")
            # print(inp)
            lime = True
        # print(type(inp), type(inp) != torch.Tensor)
        if len(self.layers) == 1 and self.layers != [0]:
            relu = nn.ReLU()

            prod_w0 = self.hidden(inp)

            hidden = relu(self.bn(prod_w0))

            w_final = self.linear(hidden)

            probs = F.softmax(w_final, dim=-1)
            res =  probs
            if lime:
                res = res.cpu()
            return res
        res =  F.softmax(self.model(inp), dim=-1)
        if lime:
            res = res.cpu()
        return res

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform(m.weight.data)
        # init.constant(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_uniform(m.weight.data)
    elif classname.find("BatchNorm2d") != -1:
        # nn.init.xavier_uniform(m.weight.data)
        # init.constant(m.bias.data, 0.0)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm1d") != -1:
        # nn.init.xavier_uniform(m.weight.data)
        # init.constant(m.bias.data, 0.0)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)