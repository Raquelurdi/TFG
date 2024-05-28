import os, glob
import numpy as np
import pickle as pkl

def get_groups(p:str, classes:list, default:str="4949", other:bool=False) -> list:
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    res = []
    for line in lines:
        try:
            l, c, ini, fin = line.strip().split(" ")
        except:
            c, ini, fin = line.strip().split(" ")
            l = default
        c_real = c
        if c not in classes:
            if not other:
                continue
            else:
                c = "other"
        ini, fin = int(ini), int(fin)
        res.append([l, c, ini, fin, c_real])
    return res

def read_tfidf_file(p:str):
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    res = {}
    for line in lines[1:]:
        word, *tfidf_v = line.strip().split(" ")
        tfidf_v = [float(x) for x in tfidf_v[:-1]]
        res[word] = tfidf_v
    return res

def create_group(l, c, ini, fin, tfidf_file:dict):
    tfidfs = []
    for i in range(ini, fin+1):
        vector_tfidf = tfidf_file.get(f'{l}_page_{i}_{c}.idx')
        if vector_tfidf is None:
            vector_tfidf = tfidf_file.get(f'page_{i}_{c}.idx')
            if vector_tfidf is None:
                raise Exception(f"vector_tfidf is None - {l}_page_{i}_{c}.idx")
        tfidfs.append(vector_tfidf)
    return np.array(tfidfs, np.float32)


def main(path_tfidf:str, path_gruos:str, path_save:str, classes:list, default:str, other:bool):
    tfidf_file = read_tfidf_file(path_tfidf)
    res_groups = get_groups(path_gruos, classes, default, other=other)
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    for l, c, ini, fin, c_real in res_groups:
        v = create_group( l, c_real, ini, fin, tfidf_file)
        path_save_f = os.path.join(path_save, f'{l}_pages_{ini}-{fin}_{c}.idx')
        print(l, c_real, ini, fin, v.shape, path_save_f)
        with open(path_save_f, 'wb') as handle:
            pkl.dump(v, handle, protocol=pkl.HIGHEST_PROTOCOL)
        


if __name__ == "__main__":
    # default="JMBD4949"
    # other=True
    # path_groups = "/data/carabela_segmentacion/JMBD4949_4950_1page_idx/groups"
    # path_tfidf = "/data2/jose/projects/docClasifIbPRIA22/work_JMBD4949_4950_loo_1page_other/tfidf_4949_4950_loo.txt"
    # path_save = "/data/carabela_segmentacion/JMBD4949_4950_1page_idx/sequence_groups"
    # classes = [x.lower() for x in "P,CP,O,A,T,V,R,CEN,DP,D,C,TH".split(",")]
    # main(path_tfidf, path_groups, path_save, classes, default,other=other)

    default="JMBD4949"
    other=True
    path_groups = "../work_JMBD4949_4950_loo_1page_other/groups"
    path_tfidf = "../work_JMBD4949_4950_loo_1page_other/tfidf_4949_4950_loo.txt"
    path_save = "../work_JMBD4949_4950_loo_1page_other/sequence_groups"
    classes = [x.lower() for x in "P,CP,O,A,T,V,R,CEN,DP,D,C,TH".split(",")]
    main(path_tfidf, path_groups, path_save, classes, default,other=other)


    # default="JMBD4949"
    # other=True
    # path_groups = "/data/carabela_segmentacion/idxs_JMBD4949/idxs_clasif_per_page/all_classes_noS/groups"
    # path_tfidf = "/data2/jose/projects/docClasifIbPRIA22/work_tr49_te50_perPage/tfidf_tr49.txt"
    # path_save = "/home/jose/projects/docClasifIbPRIA22/work_tr49_te50_perPage/sequence_groups_tr49"
    # classes = [x.lower() for x in "P,CP,O,A,T,V,R,CEN,DP,D,C,TH".split(",")]
    # main(path_tfidf, path_groups, path_save, classes, default,other=other)

    # default="JMBD4950"
    # other=True
    # path_groups = "/data/carabela_segmentacion/idxs_JMBD4950/idxs_clasif_per_page/all_classes_noS/groups"
    # path_tfidf = "/data2/jose/projects/docClasifIbPRIA22/work_tr49_te50_perPage/tfidf_te50.txt"
    # path_save = "/home/jose/projects/docClasifIbPRIA22/work_tr49_te50_perPage/sequence_groups_te50"
    # classes = [x.lower() for x in "P,CP,O,A,T,V,R,CEN,DP,D,C,TH".split(",")]
    # main(path_tfidf, path_groups, path_save, classes, default,other=other)

    # default="JMBD4949"
    # other=True
    # path_groups = "/data/carabela_segmentacion/idxs_JMBD4949/idxs_clasif_per_page/all_classes_noS/groups"
    # path_tfidf = "/data2/jose/projects/docClasifIbPRIA22/work_tr49_te50_groups/tfidf_te50.txt"
    # path_save = "/data/carabela_segmentacion/works_tr49_te50_groups/sequence_groups_te50"
    # classes = [x.lower() for x in "P,CP,O,A,T,V,R,CEN,DP,D,C,TH".split(",")]
    # main(path_tfidf, path_groups, path_save, classes, default,other=other)