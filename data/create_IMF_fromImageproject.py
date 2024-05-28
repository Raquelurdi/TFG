import glob, os

def make_structure(path_res:str):
    if not os.path.exists(path_res):
        os.mkdir(path_res)
    tr, te = os.path.join(path_res, "train"), os.path.join(path_res, "test")
    if not os.path.exists(tr):
        os.mkdir(tr)
    if not os.path.exists(te):
        os.mkdir(te)
    return tr, te

def prepare_save_idx_reduced(p:str, p_dest:str):
    f = open(p_dest, "w")
    f_orig = open(p, "r")
    for line in f_orig.readlines():
        line = line.strip().split(" ")
        word, prob = line[0], float(line[2])
        f.write(f'{word} {prob}\n')
    f_orig.close()
    f.close()

def main(path:str, path_save:str, path_idx_tr:str, path_idx_te:str):
    # files = glob.glob(os.path.join(path, "*"))
    tr_path, te_path = make_structure(path_save)
    files_tr_f = glob.glob(os.path.join(path, "train/F/*"))
    files_tr_f.extend(glob.glob(os.path.join(path, "validation/F/*")))
    files_tr_m = glob.glob(os.path.join(path, "train/M/*"))
    files_tr_m.extend(glob.glob(os.path.join(path, "validation/M/*")))
    files_tr_i = glob.glob(os.path.join(path, "train/I/*"))
    files_tr_i.extend(glob.glob(os.path.join(path, "validation/I/*")))
    print(len(files_tr_i),len(files_tr_m), len(files_tr_f))
    arr = [("F", files_tr_f), ("M", files_tr_m), ("I", files_tr_i)]
    for l, files_tr in arr:
        for file_tr in files_tr:
            fname = file_tr.split("/")[-1].split(".")[0]
            fname_path = os.path.join(path_idx_tr, f"{fname}.idx")
            fname_path_dest = os.path.join(tr_path, f"{fname}_{l}.idx")
            # print(fname, fname_path, fname_path_dest)
            if not os.path.exists(fname_path_dest):
                prepare_save_idx_reduced(fname_path, fname_path_dest)

    files_te_f = glob.glob(os.path.join(path, "test/F/*"))
    files_te_m = glob.glob(os.path.join(path, "test/M/*"))
    files_te_i = glob.glob(os.path.join(path, "test/I/*"))
    print(len(files_te_i),len(files_te_m), len(files_te_f))
    arr = [("F", files_te_f), ("M", files_te_m), ("I", files_te_i)]
    for l, files_te in arr:
        for file_te in files_te:
            fname = file_te.split("/")[-1].split(".")[0]
            fname_path = os.path.join(path_idx_te, f"{fname}.idx")
            fname_path_dest = os.path.join(te_path, f"{fname}_{l}.idx")
            # print(fname, fname_path, fname_path_dest)
            if not os.path.exists(fname_path_dest):
                prepare_save_idx_reduced(fname_path, fname_path_dest)


if __name__ == "__main__":
    tr = "tr49"
    path = f"/home/jose/projects/image_classif/data/JMBD4949_4950/{tr}"
    path_save = f"/home/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/{tr}"
    if tr == "tr49":
        path_idx_tr = "/data/carabela_segmentacion/idxs_JMBD4949/JMBD_4949"
        path_idx_te = "/data/carabela_segmentacion/idxs_JMBD4950/JMBD_4950"
    else:
        path_idx_te = "/data/carabela_segmentacion/idxs_JMBD4949/JMBD_4949"
        path_idx_tr = "/data/carabela_segmentacion/idxs_JMBD4950/JMBD_4950"
    main(path, path_save, path_idx_tr, path_idx_te)