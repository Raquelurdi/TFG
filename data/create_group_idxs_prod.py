import os, glob

def save_idxs(idxs, path_save):
    f = open(path_save, "w")
    for word, prob in idxs:
        f.write("{} {}\n".format(word,prob))
    f.close()

def concat_idxs(idxs:list) -> list:
    res = []
    for fpath_idx in idxs:
        f = open(fpath_idx, "r")
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.strip()
            s = line.split(" ")
            word = s[0]
            prob = float(s[1])
            res.append((word, prob))
    return res

def read_file_groups(p:str, idxs_paths:dict) -> list:
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    res = []
    for line in lines:
        ini, fin, *err_type = line.strip().split()
        ini, fin = int(ini), int(fin)
        print(ini, fin)
        files = []
        for i in range(ini, fin+1):
            files.append(idxs_paths[i])
        res.append((ini,fin, files))
    return res

def read_files(p:str) -> dict:
    res = {}
    all = glob.glob(os.path.join(p, "*idx"))
    for f in all:
        page = int(f.split("/")[-1].split("_")[1])
        res[page] = f
    return res

def main(path_idxs_per_page:str, path_file_group:str, path_save_prod:str):
    ## Create idxs - train and test
    if not os.path.exists(path_save_prod):
        os.mkdir(path_save_prod)
    idxs_paths = read_files(path_idxs_per_page)
    res_groups = read_file_groups(path_file_group, idxs_paths)
    for ini, fin, group in res_groups:
        group_idxs = concat_idxs(group)
        tip = "production"
        path_file = os.path.join(path_save_prod, "pages_{}-{}_{}.idx".format(ini, fin, tip))
        save_idxs(group_idxs, path_file)

if __name__ == "__main__":
    path_idxs_per_page = "/data/carabela_segmentacion/idxs_JMBD4950/idxs_clasif_per_page/all_classes_noS"
    path_file_group = "/data2/jose/projects/docClasifIbPRIA22/work_tr49_te50_groups/file_te50_badsegments_gt"
    path_save_prod = "/data2/jose/projects/docClasifIbPRIA22/work_tr49_te50_groups/prod_files_idxs"
    main(path_idxs_per_page, path_file_group, path_save_prod)