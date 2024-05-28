from __future__ import print_function
from __future__ import division
from builtins import range
import numpy as np
import os
from math import sqrt
from sklearn.metrics import confusion_matrix
import pandas as pd

# --- Lavenshtein edit distance
def levenshtein(hyp, target):
    """
    levenshtein edit distance using
    addcost=delcost=subcost=1
    Borrowed form: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    if len(hyp) < len(target):
        return levenshtein(target, hyp)

    # So now we have len(hyp) >= len(target).
    if len(target) == 0:
        return len(hyp)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    hyp = np.array(tuple(hyp))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in hyp:
        # Insertion (target grows longer than hyp):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and hyp items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
            current_row[1:], np.add(previous_row[:-1], target != s)
        )

        # Deletion (target grows shorter than hyp):
        current_row[1:] = np.minimum(current_row[1:], current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

def read_classes(p:str):
    f = open(p, "r")
    l = f.readlines()
    f.close()
    return {int(x.strip().split(" ")[1]):x.strip().split(" ")[0] for x in l}, [x.strip().split(" ")[0] for x in l]

def get_acc_file(p:str, dict_classes:dict, classes:list):
    f = open(p, "r")
    lines = f.readlines()[1:]
    f.close()
    gts, hyp_l = [], []
    gts_c, hyp_l_c = [], []
    errors = []
    for line in lines:
        line = line.strip().split(" ")
        l = line[0]
        gt = int(line[1])
        hyps = [float(x) for x in line[2:]]
        h = np.argmax(hyps)
        gts.append(gt)
        hyp_l.append(h)
        gts_c.append(dict_classes.get(gt))
        hyp_l_c.append(dict_classes.get(h))
        if h != gt:
            errors.append((l, dict_classes.get(gt), dict_classes.get(h), max(hyps)))
    gts = np.array(gts)
    hyp_l = np.array(hyp_l)
    gts_c = np.array(gts_c)
    hyp_l_c = np.array(hyp_l_c)
    acc = (gts==hyp_l).sum() / len(gts)
    acc *= 100.0
    interval = (1.96 * sqrt((acc/100*(1-acc/100))/len(gts)))*100

    ## Confmat
    confmat = confusion_matrix(gts_c, hyp_l_c, labels=classes)
    priors = np.unique(gts, return_counts=True)

    return acc, interval, confmat, priors, errors

if __name__ == "__main__":
    p = "/data2/jose/projects/docClasifIbPRIA22/works_LOO_JMBD4949_allFiles"
    path_classes_dict = "/home/jose/projects/docClasifIbPRIA22/work_JMBD4949_loo_allFiles/tfidf_4949_loo_classes.txt"
    dict_classes, classes = read_classes(path_classes_dict)
    first = True
    index = [f'true:{x}' for x in classes]
    columns = [f'pred:{x}' for x in classes]
    work="128,128"
    paths = [
        f'work_{work}_numFeat256',
        f'work_{work}_numFeat512',
        f'work_{work}_numFeat1024',
        f'work_{work}_numFeat2048',
        f'work_{work}_numFeat4096',
        f'work_{work}_numFeat8192',
        f'work_{work}_numFeat16384',        
        ]
    # for d in list(os.walk(p))[0][1]:
    for d in paths:
        if not d:
            continue
        path = os.path.join(p, d, "results.txt")
        acc, interval, confmat, priors, errors = get_acc_file(path, dict_classes, classes)
        if first:
            cs, totals = priors
            total = np.sum(totals)
            totals_ = totals / total
            print("#"*5)
            for c, t_, t in zip(cs, totals_, totals):
                print(f'Class {dict_classes.get(c)} - {t_} ({t}/{total})')
            print("#"*5)
            first = False
        print("---"*10)
        print(f'File {d} : Accuracy {acc}  Error {100-acc}  +-{interval}')
        cmtx = pd.DataFrame(
                confmat, 
                index=index, 
                columns=columns
            )
        print(cmtx)
        for l,gt ,h, prob  in errors:
            print(f'       Error in file {l}   GT {gt}  HYP {h} (Prob {prob*100.})')
