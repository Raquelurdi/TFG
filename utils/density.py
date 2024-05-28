import glob, os


def load_idxs(path_all:str, min_rw:float):
    all_paths = glob.glob(os.path.join(path_all, "*idx"))
    n_X = 0
    nspots = 0
    for path in all_paths:
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        for line in lines:
            try:
                word, prob = line.strip().split(" ")
                prob = float(prob)
            except:
                continue
            if prob > min_rw:
                n_X += prob
                nspots += 1
    return n_X, nspots
            

def main(path_idxs:str, min_rw:float):
    n_X, nspots = load_idxs(path_idxs, min_rw=0)
    density = nspots / n_X
    print(f"Density for minRP = 0: {density} [n_X {n_X}  |    nspots {nspots}]")
    n_X, nspots = load_idxs(path_idxs, min_rw)
    density = nspots / n_X
    print(f"Density for minRP = {min_rw}: {density} [n_X {n_X}  |    nspots {nspots}]")

if __name__ == "__main__":
    path_idxs = "PrIx/JMBD4949_4950_idx"
    min_rw = 0.5
    main(path_idxs, min_rw)