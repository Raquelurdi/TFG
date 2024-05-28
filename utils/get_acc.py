import numpy as np 
from math import sqrt

def read_results(p:str):
    f = open(p, "r")
    lines = f.readlines()[1:]
    f.close()
    res = {}
    hyps_, gts_ = [], []
    pnames = []
    max_h = []
    for line in lines:
        pname, gt, *hyps = line.strip().split(" ")
        gt = int(gt)
        hyps = [float(x) for x in hyps]
        pages = pname.split("_")[2]
        l = pname.split("_")[0]
        res[f'{l}_{pages}'] = [gt,hyps]
        h = np.argmax(hyps)
        max_h.append(hyps[h])
        # print(hyps[h])
        hyps_.append(h)
        gts_.append(gt)
        pnames.append(pname)
    max_h = np.array(max_h)
    # print("-> mean", np.mean(max_h))
    # print(np.sum(max_h > 0.99))

    return res, hyps_, gts_, pnames

if __name__ == "__main__":
    # path_results = "/data2/jose/projects/docClasifIbPRIA22/works_JMBD4949_loo_1page_LSTM/work_128,128_numFeat1024_128epochs_0.01lrADAM/results.txt"
    tr = "tr49"; te = "te50"
    # tr = "tr50"; te = "te49"
    nmb_feats = [2**x for x in range(4,15)]
    layers_list=["128,128"]
    # work_dir = f"works_{tr}_{te}_groups_11classes_other"

    work_dir = f"works_JMBD4949_4950_loo_groups" # 12 clases
    print(work_dir)
    print_as_table = True
    if print_as_table:
        print(f"|V|    \t {' '.join(layers_list)}")
    for feats in nmb_feats:
        if not print_as_table:
            print(f"TRAINING WITH {tr} - layers {layers}")
        r = ""
        for layers in layers_list:
            
            path_results = f"../{work_dir}/work_{layers}_numFeat{feats}/results.txt"
            try:
                res, hyps, gts, pnames = read_results(path_results)
                gts = np.array(gts)
                hyps = np.array(hyps)
                errors = (gts == hyps).sum()
                acc = errors / hyps.shape[0]
                e = (1-acc)*100
                dataset_total = len(hyps)
                interval = (1.96 * sqrt((e/100*(1-e/100))/dataset_total))*100
                if not print_as_table:
                    print(f'{feats} feats Error {e} from {len(hyps)} samples - [{len(hyps) - errors} errors]')
                else:
                    r = f"{r}\t{e:.2f} [{interval:.2f}]"
                
            except Exception as e:
                if not print_as_table:
                    print(f'{feats} not found [{path_results}]')
                else:
                    r = f"{r}\t-"


        if print_as_table:
                print(f"{feats}\t{r}")
    
