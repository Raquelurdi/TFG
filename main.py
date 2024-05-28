from __future__ import print_function
from __future__ import division
import time
from unittest import result
import torch.optim as optim
import torch
import torch.nn as nn
import logging, os
import numpy as np
from utils.optparse import Arguments as arguments
torch.autograd.set_detect_anomaly(True)
from torch.autograd import Variable
from data import dataset
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
import wandb
from utils.voting import voting
# from torchvision import transforms
from data import transforms as data_transforms

torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def acc_on_nbest(gts, hyps, n=5):
    res = []
    for i, _ in enumerate(gts):
        gt = gts[i]
        hyp = hyps[i]
        best_n = hyp.argsort()[-n:]
        #print(gt, best_n, gt in best_n, "({})".format(n))
        res.append(gt in best_n)
    return res, np.mean(res)*100

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results_eval(results_tests, dir, logger, number_to):
    create_dir(dir)
    fname = os.path.join(dir, "results_eval")
    logger.info("Saving results on {}".format(fname))
    f = open(fname, "w")
    #f.write("ID-testSample ID-Class GT(0|1)  P(Class|testSample)\n")
    for id_x, label, prediction in results_tests:
        #c, pages = id_x.split("_")
        for i, p_hyp in enumerate(prediction):
            gt_01 =  int(i == label)
            c = number_to[i]
            f.write("{} {} {} {}\n".format(id_x, c, gt_01, p_hyp))
    f.close()

def save_results(dataset, tensor, opts, _name="", ids=None, ys=None):
    outputs = tensor_to_numpy(tensor)
    class_dict, number_to_class = load_dict_class(opts.class_dict)
    dir = opts.work_dir
    create_dir(dir)
    fname = os.path.join(dir, f"results{_name}.txt")
    f = open(fname, "w")
    last_layer = "Softmax"
    if opts.openset == "onevsall":
        last_layer = "Sigmoid"
    f.write(f"Legajo GT(index) {last_layer}")
    for i in range(len(number_to_class)):
        f.write(f' {number_to_class[i]}')
    f.write("\n")
    if dataset is not None:
        ids = dataset.ids
        ys = [y[1] for y in dataset.data]

    sum_ = 0
    for id_x, label, prediction in zip(ids, ys, outputs):
        res=""
        p = np.argmax(prediction)
        sum_ += label != p
        # prediction[label] = 1
        for s in prediction:
            res+=" {}".format(str(s))
        f.write("{} {}{}\n".format(id_x, label, res))
    f.close()
    print(f"Error save_results {sum_} errors - {(sum_/len(outputs))*100.0}")

def save_results_per_class(results_tests, dir, logger):
    create_dir(dir)
    fname = os.path.join(dir, "results_per_class.txt")
    logger.info("Saving results per class on on {}".format(fname))
    f = open(fname, "w")
    for line in results_tests:
        f.write("{}\n".format(line))
    f.close()

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def to_one_hot(y, n_dims=7):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    y =  Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot
    return torch.transpose(torch.transpose(y, 1,3), 2,3)

def prepare():
    """
    Logging and arguments
    :return:
    """

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    # --- keep this logger at DEBUG level, until aguments are processed
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --- Get Input Arguments
    in_args = arguments(logger)
    opts = in_args.parse()
    # if check_inputs_graph(opts, logger):
    #     logger.critical("Execution aborted due input errors...")
    #     exit(1)

    fh = logging.FileHandler(opts.log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # --- restore ch logger to INFO
    ch.setLevel(logging.INFO)
    return logger, opts

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

def save_results_train(textDataset, y_train, results_train, opts, last_layer_str, fname, logger):
    path = os.path.join(opts.work_dir, "results_train")
    if not os.path.exists(path):
        os.mkdir(path)
    path_file = os.path.join(path, fname)
    f = open(path_file, "w")
    f.write(f"{last_layer_str}\n")
    ids = textDataset.cancerDt_val.ids
    results_train = tensor_to_numpy(results_train)
    # print("ids", ids)
    # print("results_train", results_train)
    # print("y_train", y_train)
    # JMBD4949_pages_1022-1023_p.idx
    acc = []
    for id_, res, y in zip(ids, results_train, y_train):
        res_str=""
        hyp = np.argmax(res)
        acc.append(y == hyp)
        for s in res:
            res_str+=" {}".format(str(s))
        s = "{} {}{}\n".format(id_, y, res_str)
        f.write(f"{s}")
    f.close()
    # print(acc)
    # print(np.sum(acc), len(acc))
    err = 1 - (np.sum(acc) / len(acc))
    logger.info(f"Error training {fname}: {err}")

def main():

    logger, opts = prepare()
    print(opts.work_dir)

    device = torch.device(f"cuda:{opts.gpu}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    torch.set_default_tensor_type("torch.FloatTensor")
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    random.seed(opts.seed)
    os.environ['PYTHONHASHSEED'] = str(opts.seed)
    net = None
    logger.info(opts)
    logger.info("Model: {}".format(opts.model))
    from models import model as models



    logger_csv = CSVLogger(opts.work_dir, name=opts.exp_name)
    
    path_save = os.path.join(opts.work_dir, "checkpoints")
    if not opts.LOO:
        wandb.init()
        wandb.run.name = opts.work_dir
        wandb.run.save()
        wandb.config = {
            "layers": ",".join([str(z) for z in opts.layers]),
            "num_feats": opts.num_feats,
        }
        wandb_logger = WandbLogger(project=opts.exp_name)
        # train_transforms = transforms.Compose([data_transforms.Addnoise(k=50)])
        # val_transforms = train_transforms
        train_transforms, val_transforms = None, None
        textDataset = dataset.TextDataset(opts=opts, train_transforms=train_transforms, val_transforms=val_transforms)
        net = models.Net(layers=opts.layers, len_feats=textDataset.len_feats, n_classes=len(opts.classes), opts=opts)
        print(net)
        if opts.checkpoint_load:
            net = net.load_from_checkpoint(opts.checkpoint_load, layers=opts.layers, len_feats=textDataset.len_feats, n_classes=len(opts.classes), opts=opts)
        net.to(device)
        wandb_logger.watch(net)
        early_stop_callback = EarlyStopping(monitor="val_epoch_loss", min_delta=0.00, patience=50, verbose=True, mode="max")
        trainer = pl.Trainer(min_epochs=20, max_epochs=opts.epochs, logger=[logger_csv, wandb_logger], #wandb_logger
                deterministic=True if opts.seed is not None else False, auto_select_gpus=True,
                default_root_dir=path_save,
                auto_lr_find=opts.auto_lr_find,
                benchmark=True,
                # amp_level="03",
                # amp_backend='apex',
                gradient_clip_val=0.2,
                # auto_scale_batch_size="power",
                callbacks=[early_stop_callback, 
                    StochasticWeightAveraging(
                        swa_epoch_start = 1,
                        # swa_lrs: Optional[Union[float, List[float]]] = None,
                        annealing_epochs= 3,
                        annealing_strategy = "cos",
                        # swa_lrs=1e-2
                    )
                ]
            )
        if opts.do_train and opts.auto_lr_find:
            logger.info("auto_lr_find ON - Tuning the model")
            trainer.tune(net, textDataset)
            logger.info("Model tunned")
        
        if opts.do_train:
            trainer.fit(net, textDataset)
            
        if opts.do_test:
            results_test = trainer.test(net, textDataset)
            logger.info(results_test)
            if "voting" in opts.model:
                results_test = trainer.predict(net, textDataset.test_dataloader())
                outputs, gts, ids = [], [], []
                for x in results_test:
                    o = tensor_to_numpy(x['outputs'])
                    outputs.extend(o)
                    # print(x['y_gt'], len(x['y_gt']))
                    # print(x['ids'], len(x['ids']))
                    # exit()
                    gts.extend(x['y_gt']) 
                    ids.extend(x['ids'])
                outputs = torch.from_numpy(np.array(outputs))
                save_results(None, outputs, opts, ids=ids, ys=gts)
            else:
                results_test = trainer.predict(net, textDataset.test_dataloader())
                results_test = torch.cat(results_test, dim=0)
                save_results(textDataset.cancerDt_test, results_test, opts,)
        if opts.do_prod:
            # results_test = trainer.test(net, textDataset)
            results_prod = trainer.predict(net, textDataset)
            results_prod = torch.cat(results_prod, dim=0)
            save_results(textDataset.cancerDt_prod, results_prod, opts, _name="_prod")
    else:
        n_test, num_exps = 0, 90000
        info = dataset.load_RWs(opts) #data_tr_dev, data_test, len_feats, num_classes, class_dict, number_to_class
        dir = opts.work_dir
        create_dir(dir)
        fname = os.path.join(dir, "results.txt")
        class_dict, number_to_class = load_dict_class(opts.class_dict)
        f = open(fname, "w")
        last_layer = "Softmax"
        if opts.openset == "onevsall":
            last_layer = "Sigmoid"
        # f.write("Legajo GT(index) Softmax\n")
        f.write(f"Legajo GT(index) {last_layer}")
        last_layer_str = f"Legajo GT(index) {last_layer}"
        for i in range(len(number_to_class)):
            f.write(f' {number_to_class[i]}')
            last_layer_str += f' {number_to_class[i]}'
        f.write("\n")
        ys, hyps = [], []
        num_exps = len(info[0]) + 1
        path_save_remove = os.path.join(path_save, "*")
        if opts.path_file_groups == "":
            while(n_test < num_exps):
                logger.info(f'Exp {n_test} \ {num_exps}')
                os.system(f'rm -rf {path_save_remove}') #TODO change
                textDataset = dataset.TextDataset(opts=opts, n_test=n_test, info=info)
                fname = textDataset.data_test[0][-1].split(".")[0]
                n_test += 1
                net = models.Net(layers=opts.layers, len_feats=textDataset.len_feats, n_classes=textDataset.num_classes, opts=opts)
                net.to(device)
                # print(net)
                early_stop_callback = EarlyStopping(monitor="val_epoch_loss", min_delta=0.00, patience=50, verbose=True, mode="max")
                trainer = pl.Trainer(min_epochs=20, max_epochs=opts.epochs,
                logger=[logger_csv], #wandb_logger
                        deterministic=True if opts.seed is not None else False,
                        default_root_dir=path_save, auto_select_gpus=True,
                        gradient_clip_val=0.2,
                        callbacks=[early_stop_callback,]
                    )
                trainer.fit(net, textDataset)
                results_test = trainer.test(net, textDataset)
                if "voting" in opts.model:
                    results_test = trainer.predict(net, textDataset.test_dataloader())
                    outputs, gts, ids = [], [], []
                    for x in results_test:
                        o = tensor_to_numpy(x['outputs'])
                        outputs.extend(o)
                        # print(x['y_gt'], len(x['y_gt']))
                        # print(x['ids'], len(x['ids']))
                        # exit()
                        gts.extend(x['y_gt']) 
                        ids.extend(x['ids'])
                    outputs = torch.from_numpy(np.array(outputs))
                    # save_results(None, outputs, opts, ids=ids, ys=gts)
                    # print(outputs.shape)
                    results_test = outputs
                else:
                    results_test = trainer.predict(net, textDataset.test_dataloader())
                    results_test = torch.cat(results_test, dim=0)

                    results_train = trainer.predict(net, textDataset.val_dataloader())
                    results_train = torch.cat(results_train, dim=0)
                    # save_results(textDataset.cancerDt_test, results_test, opts,)
                # results_test = trainer.predict(net, textDataset.test_dataloader())
                
                # results_test = torch.cat(results_test, dim=0)

                # Save to file
                # print([y[1] for y in textDataset.cancerDt_test.data])
                y = [y[1] for y in textDataset.cancerDt_test.data][0]
                # print(y)
                save_to_file(textDataset, f, y, results_test, opts)
                ys.append(y)
                hyps.append(np.argmax(results_test))

                if not "voting" in opts.model:
                    y_train = [y[1] for y in textDataset.cancerDt_val.data]
                    save_results_train(textDataset, y_train, results_train, opts, last_layer_str, fname, logger)
                del net
        else:
            groups = get_groups(opts.path_file_groups, opts.classes)
            for ngroup, (l, c, ini, fin) in enumerate(groups):
                os.system(f'rm -rf {path_save_remove}') #TODO change
                print(f'Group {ngroup}/{len(groups)} {c} {ini} {fin} {l}')
                for npage in range(ini, fin+1):
                    n_test = search_page(info[0], npage, l)
                    textDataset = dataset.TextDataset(opts=opts, n_test=n_test, info=info, legajo=l)
                    n_test += 1
                    logger.info(f'page: {npage} (num {n_test} in data) - {l}')
                    if npage == ini:
                        print(f'Training for the first time')
                        
                        net = models.Net(layers=opts.layers, len_feats=textDataset.len_feats, n_classes=textDataset.num_classes, opts=opts)
                        net.to(device)
                        early_stop_callback = EarlyStopping(monitor="val_epoch_loss", min_delta=0.00, patience=50, verbose=True, mode="max")
                        trainer = pl.Trainer(min_epochs=20, max_epochs=opts.epochs, logger=[logger_csv], #wandb_logger
                            deterministic=True if opts.seed is not None else False,
                            default_root_dir=path_save, auto_select_gpus=True,
                            gradient_clip_val=0.2,
                            callbacks=[early_stop_callback,]
                        )
                        try:
                            trainer.fit(net, textDataset)
                        except Exception as e:
                            print(f'Problem with sample page: {npage} (num {n_test} in data) - {l}')
                            # print(f'{net}')
                            raise e
                    else:
                        print(f'Using already trained model')
                    results_test = trainer.test(net, textDataset)
                    results_test = trainer.predict(net, textDataset.test_dataloader())
                    results_test = torch.cat(results_test, dim=0)
                    # Save to file
                    y = [y[1] for y in textDataset.cancerDt_test.data][0]
                    save_to_file(textDataset, f, y, results_test, opts)
                    ys.append(y)
                    hyps.append(np.argmax(results_test))
                del net
                print("--------------------\n\n")
            # acc_v, acc_results, fallos = voting(read_results(fname), groups)
            # logger.info(f'Accuracy voting: {acc_v}')
            # logger.info(f'Error voting: {1-acc_v}')
        f.close()
        acc = accuracy_score(ys, hyps)
        logger.info(f'Accuracy: {acc}')
        logger.info(f'Error: {1-acc}')
        
def read_results(p:str):
    f = open(p, "r")
    lines = f.readlines()[1:]
    f.close()
    res = {}
    for line in lines:
        pname, gt, *hyps = line.strip().split(" ")
        gt = int(gt)
        hyps = [float(x) for x in hyps]
        page = int(pname.split("_")[2])
        res[page] = [gt,hyps]
    return res    

def search_page(data:list, num, l:str):
    for i, d in enumerate(data):
        npage = int(d[-1].split("_")[2])
        l_page = d[-1].split("_")[0]
        if npage == num and l_page == l:
            return i 
    raise Exception(f'page for {num} not found')

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

def save_to_file(textDataset, f, y, results_test, opts):
    ids = textDataset.cancerDt_test.ids[0]
    # results_test = tensor_to_numpy(results_test)[0]
    # print("save to file ", opts.model)
    if "voting" not in opts.model:
        results_test = tensor_to_numpy(results_test)[0]
        res=""
        for s in results_test:
            res+=" {}".format(str(s))
        s = "{} {}{}\n".format(ids, y, res)
        print(s)
        f.write(s)
    else:
        results_test = tensor_to_numpy(results_test)
        # JMBD4949_pages_1022-1023_p.idx
        l, _, pages, c = ids.split("_")
        ini, fin = pages.split("-")
        ini, fin = int(ini), int(fin)
        j = 0
        for i in range(ini, fin+1):
            results_i = results_test[j]
            # print(l, i, results_i)
            #JMBD4950_page_4_cp.idx 1 0.20062275 0.17206226 0.15092145 0.049790654 0.12978472 0.12793754 0.006852842 0.078733616 0.01871656 0.007054133 0.017348334 0.016604096 0.023571102
            id_ = f'{l}_page_{i}_{c}'
            j += 1
            res=""
            for s in results_i:
                res+=" {}".format(str(s))
            s = "{} {}{}\n".format(id_, y, res)
            f.write(s)
    f.flush()
    

if __name__ == "__main__":
    main()