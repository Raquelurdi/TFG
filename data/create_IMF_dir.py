import cv2
import glob, os
from sklearn.utils import shuffle
import tqdm

def read_csv(fpath, ktype):
    f = open(fpath, "r", encoding='utf-8', errors='ignore')
    if ktype == "49":
        lines = f.readlines()[:-1]
    else:
        lines = f.readlines()[1:]
    f.close()
    res = []
    for i, line in enumerate(lines):
        # print("line ", i+1)
        if ktype == "49":
            line = line.split(",")
            p_ini = line[1]
            p_fin = line[2]
            tip_abreviada = line[4].lower()
        else:
            line = line.split(",")
            p_ini = line[0]
            p_fin = line[1]
            tip_abreviada = line[3].lower()
        p_ini = int(p_ini)
        p_fin = int(p_fin)
        
        tip_abreviada = tip_abreviada.replace(" ", "")
        tip_abreviada = tip_abreviada.replace("?", "unk")
        # tip_abreviada = tip_abreviada.replace("s?", "s")
        # if tip_abreviada == "?":
        #     continue
        if "-" in tip_abreviada:
            tip_abreviada = tip_abreviada.split("-")[0]
        if tip_abreviada == "s":
            tip_abreviada = "p"
        res.append((tip_abreviada, p_ini, p_fin))
    return res

def init_JMBD(csv_path, idx_dirs, corpus):
    if corpus == "JMBD4949":
        ktype = "49"
    else:
        ktype = "50"
    files = read_csv(csv_path, ktype=ktype)
    res = []
    for tip_abreviada, p_ini, p_fin in files:
        for num in range(p_ini, p_fin+1):
            if ktype == "49":
                fpath_img = os.path.join(idx_dirs, f"JMBD_4949_{num:05}.idx")
            else:
                fpath_img = os.path.join(idx_dirs, f"JMBD_4950_{num:05}.idx")
            if num == p_ini:
                c = "I"
            elif num == p_fin:
                c = "F"
            else:
                c = "M"
            res.append((fpath_img, c))
    return res

def create_structure(path_res, classes):
    if not os.path.exists(path_res):
        os.mkdir(path_res)
    tr, te, val = os.path.join(path_res, "train"), os.path.join(path_res, "test"), os.path.join(path_res, "validation")
    if not os.path.exists(tr):
        os.mkdir(tr)
    if not os.path.exists(te):
        os.mkdir(te)
    if not os.path.exists(val):
        os.mkdir(val)
    for c in classes:
        for t in [tr,te,val]:
            p = os.path.join(t, c)
            if not os.path.exists(p):
                os.mkdir(p)

def create_structure_crossBundles(path_res, classes):
    if not os.path.exists(path_res):
        os.mkdir(path_res)
    res = []
    for t in ["tr49", "tr50"]:
        path_res_ = os.path.join(path_res, t)
        res.append(path_res_)
        if not os.path.exists(path_res_):
            os.mkdir(path_res_)
        print(path_res_)
        tr, te, val = os.path.join(path_res_, "train"), os.path.join(path_res_, "test"), os.path.join(path_res_, "validation")
        if not os.path.exists(tr):
            os.mkdir(tr)
        if not os.path.exists(te):
            os.mkdir(te)
        if not os.path.exists(val):
            os.mkdir(val)
        for c in classes:
            for t in [tr,te,val]:
                p = os.path.join(t, c)
                if not os.path.exists(p):
                    os.mkdir(p)
    return res

def load(fpath, height, width):
    img = cv2.imread(fpath)
    if height is not None and width is not None:
        img = cv2.resize(img, (width, height)) 
    return img

def create_data(tr_data, te_data, val_data, path_res):
    for fpath_img, c in tqdm.tqdm(tr_data):
        path_folder = os.path.join(path_res, "train", c)
        fname = fpath_img.split("/")[-1].split(".")[0]
        path_save = os.path.join(path_folder, fname+".jpg")
        img = load(fpath_img, width=width, height=height)
        cv2.imwrite(path_save, img)
    for fpath_img, c in tqdm.tqdm(te_data):
        path_folder = os.path.join(path_res, "test", c)
        fname = fpath_img.split("/")[-1].split(".")[0]
        path_save = os.path.join(path_folder, fname+".jpg")
        img = load(fpath_img, width=width, height=height)
        cv2.imwrite(path_save, img)
    for fpath_img, c in tqdm.tqdm(val_data):
        path_folder = os.path.join(path_res, "validation", c)
        fname = fpath_img.split("/")[-1].split(".")[0]
        path_save = os.path.join(path_folder, fname+".jpg")
        img = load(fpath_img, width=width, height=height)
        cv2.imwrite(path_save, img)

if __name__ == "__main__":
    corpus = "JMBD4950"
    data="all"
    classes = ["I", "M", "F"]
    # img_dirs = "/data/carabela_segmentacion/idxs_JMBD4949/JMBD_4949"
    img_dirs = "/data/carabela_segmentacion/idxs_JMBD4950/images"
    # csv_path = "/data/carabela_segmentacion/csv_gt/JMBD_4949_Clasificacion_20220405_2.csv"
    csv_path = "/data/carabela_segmentacion/csv_gt/JMBD_4950_Clasificacion_20220405.csv"
    
    auto_split = [0.60,0.10,0.30] # tr val test
    width, height = 768, 1024
    random_state=0
    
    
    if "JMBD" in corpus:
        if data == "all":
            path_res = "/home/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950"
            idx_dirs = "/data/carabela_segmentacion/idxs_JMBD4949/JMBD_4949"
            csv_path = "/data/carabela_segmentacion/csv_gt/JMBD_4949_Clasificacion_20220405_2.csv"
            res4949 = init_JMBD(csv_path=csv_path, idx_dirs=idx_dirs, corpus="JMBD4949")
            idx_dirs = "/data/carabela_segmentacion/idxs_JMBD4950/JMBD_4950"
            csv_path = "/data/carabela_segmentacion/csv_gt/JMBD_4950_Clasificacion_20220405.csv"
            res4950 = init_JMBD(csv_path=csv_path, idx_dirs=idx_dirs, corpus="JMBD4950")
            res4949 = shuffle(res4949, random_state=random_state)
            res4950 = shuffle(res4950, random_state=random_state)
            print("inacabado")
            exit()
            create_structure_crossBundles(path_res, classes)
            auto_split = 0.85 # tr val

            path_res_ = os.path.join(path_res, "tr49")
            num_tr = int(len(res4949) * auto_split)
            tr_data = res4949[:num_tr]
            val_data = res4949[num_tr:]
            te_data = res4950
            create_data(tr_data, te_data, val_data, path_res_)

            path_res_ = os.path.join(path_res, "tr50")
            num_tr = int(len(res4950) * auto_split)
            tr_data = res4950[:num_tr]
            val_data = res4950[num_tr:]
            te_data = res4949
            create_data(tr_data, te_data, val_data, path_res_)

        else:
            path_res = "/home/jose/projects/image_classif/data/JMBD4950"
            create_structure(path_res, classes)
            res = init_JMBD(csv_path=csv_path, img_dirs=img_dirs, corpus=corpus)
            if auto_split is not None:
                num_tr = int(len(res) * auto_split[0])
                num_val = int(len(res) * auto_split[1])
                num_test = len(res) - num_tr - num_val
                res = shuffle(res, random_state=random_state)
                tr_data = res[:num_tr]
                val_data = res[num_tr:num_tr+num_val]
                te_data = res[num_tr+num_val:]
                print(num_tr, num_val, num_test, len(tr_data), len(val_data), len(te_data))
                create_data(tr_data, te_data, val_data, path_res)
                # for fpath_img, c in tqdm.tqdm(tr_data):
                #     path_folder = os.path.join(path_res, "train", c)
                #     fname = fpath_img.split("/")[-1].split(".")[0]
                #     path_save = os.path.join(path_folder, fname+".jpg")
                #     img = load(fpath_img, width=width, height=height)
                #     cv2.imwrite(path_save, img)
                # for fpath_img, c in tqdm.tqdm(te_data):
                #     path_folder = os.path.join(path_res, "test", c)
                #     fname = fpath_img.split("/")[-1].split(".")[0]
                #     path_save = os.path.join(path_folder, fname+".jpg")
                #     img = load(fpath_img, width=width, height=height)
                #     cv2.imwrite(path_save, img)
                # for fpath_img, c in tqdm.tqdm(val_data):
                #     path_folder = os.path.join(path_res, "validation", c)
                #     fname = fpath_img.split("/")[-1].split(".")[0]
                #     path_save = os.path.join(path_folder, fname+".jpg")
                #     img = load(fpath_img, width=width, height=height)
                #     cv2.imwrite(path_save, img)
            