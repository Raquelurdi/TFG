from __future__ import print_function
from __future__ import division

import os, glob
import math
import operator
import argparse
import numpy as np

def _str_to_bool(data):
    """
    Nice way to handle bool flags:
    from: https://stackoverflow.com/a/43357954
    """
    if data.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif data.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser(description='Create the spans')
parser.add_argument('--prob', type=float, help='Filtering probability')
parser.add_argument('--classes', type=str, help='List of separated value of classes')
parser.add_argument('--data_path', type=str, help='Data path')
parser.add_argument('--path_res', type=str, help='Data path')
parser.add_argument('--all_files', type=_str_to_bool, help='Data path')

args = parser.parse_args()

if __name__ == "__main__":
    prob = args.prob
    clases = args.classes
    clases = list(set(clases.split(',')))#Class list
    clases = [c.lower() for c in clases]
    if args.all_files:
        clases.append("other")
    print(clases)
    print("CARGANDO TODOS LOS ARCHIVOS Y SUS PALABRAS")
    #carpeta = location of JMBD files
    carpeta = args.data_path
    # directorio  = os.listdir(carpeta)
    directorio = glob.glob(os.path.join(carpeta, "*idx"))
    m = [] #All docs list
    s = {} #Dictionary key -> doc, word, valor -> prob max
    lw_all_total = [] #Set of words
    for path in directorio:
        # path = carpeta + '/' + doc
        doc = path.split("/")[-1]
        t_doc =  doc.split('_')[-1].split(".")[0].lower()
        if t_doc not in clases and not args.all_files: 
            continue
        f = open(path, "r")
        lines = f.readlines() 
        f.close()  
       
        m.append(doc)
        for line in lines:
            line = line.strip() #Quitar espacios blancos inecesarios
            try:
                word, prob_word = line.split() #Split por espacios en blanco
            except:
                continue
            if len(word) < 3: continue
            prob_word = float(prob_word) #Cogemos la probabilidad
            if(prob_word > prob):#Posibilidad de filtrado
                lw_all_total.append(word)
                #Calculo de f(tv)
                if s.get((doc,word), 0) == 0: #Devuelve el valor de doc,word que es una prob si ya esta y si no un 0.
                    s[doc,word] = prob_word
                else:
                    s[doc,word] = max(s[doc,word], prob_word)
    lw_all_total = set(lw_all_total)
    f_tv = {} #Diccionario key -> word, valor -> estimación del numero de veces que aparece una palabra v en un doc.
    for word in lw_all_total:
        for doc in m:
            if (doc,word) in s:
                if f_tv.get(word,0) == 0:
                    f_tv[word] = float(s[doc,word])
                else:
                    f_tv[word] += float(s[doc,word])
                    
    ###Finalizamos cálculo de P(tv) y P(no_tv):
    p_tv = {} #Diccionario key -> palabra, valor -> prob que algun doc contenga v
    p_notv = {}#Diccionario key -> palabra, valor -> prob que algun doc NO contenga v
    for word in f_tv:
        p_tv[word] = f_tv[word] / len(m)
        p_notv[word] = 1 - p_tv[word]

    #1º.Para cada clase que tengamos en la lista, calculamos f(c,tv)
    f_c_tv = {} #Diccionario key -> [clase,palabra], valor ->  valor -> nº de docs de la clase
    p_c_tv = {} #Diccionario key ->(clase,palabra), valor->prob. de que algun doc sea de la clase c, estando v
    p_c_notv = {} #Diccionario key ->(clase,palabra), valor->prob. de que algun doc NO sea de la clase c, estando v
    s_c = {} #Diccionario key -> doc, palabra, valor -> prob max de esa palabra en ese doc
    v = [] #Set de palabras
    m_c = {} #Diccionario con el número de docs por clase
    
    for c in clases:
        r = 0           
        for doc in m:
            clas_doc =  doc.split('.')[0].split('_')[-1].lower()
            if clas_doc == c:
                path = carpeta + '/' + doc
                f = open(path, "r")
                lines = f.readlines() 
                f.close()
                r += 1 
                for line in lines:
                    line = line.strip() #Quitar espacios blancos inecesarios
                    if len(line.split()) < 2:
                        continue
                    word = line.split()[0] #Split por espacios en blanco
                    if len(word) < 3: continue
                    prob_word = float(line.split()[1]) #Cogemos la probabilidad
                    if(prob_word > prob):
                        v.append(word)
                        if s_c.get((c,doc,word), 0) == 0: #Devuelve el valor de doc,word que es una prob si ya esta y si no un 0.
                            s_c[c,doc,word] = prob_word
                        else:
                            s_c[c,doc,word] = max(s_c[c,doc,word], prob_word)
        m_c[c] = r       
    #### Caso especial para usar otra clase extra
    if args.all_files:
        c = "other"
        r = 0
        for doc in m:
            clas_doc =  doc.split('.')[0].split('_')[-1].lower()
            if clas_doc not in clases: # SI no ha sido tratado ya
                path = carpeta + '/' + doc
                f = open(path, "r")
                lines = f.readlines() 
                f.close()
                r += 1 
                for line in lines:
                    line = line.strip() #Quitar espacios blancos inecesarios
                    if len(line.split()) < 2:
                        continue
                    word = line.split()[0] #Split por espacios en blanco
                    if len(word) < 3: continue
                    prob_word = float(line.split()[1]) #Cogemos la probabilidad
                    if(prob_word > prob):
                        v.append(word)
                        if s_c.get((c,doc,word), 0) == 0: #Devuelve el valor de doc,word que es una prob si ya esta y si no un 0.
                            s_c[c,doc,word] = prob_word
                        else:
                            s_c[c,doc,word] = max(s_c[c,doc,word], prob_word)
        m_c[c] = r 
    #### FIN Caso especial     
    v = set(v)
    for c in clases:
        for doc in m:
            for word in v:
                if (c,doc,word) in s_c:   
                    if f_c_tv.get((c,word),0) == 0:
                        f_c_tv[c,word] = float(s_c[c,doc,word])
                    else:
                        f_c_tv[c,word] += float(s_c[c,doc,word])
    #### Caso especial para usar otra clase extra
    # if args.all_files:
    #     v = "other"
    #     for doc in m:
    #         for word in v:
    #             if (c,doc,word) in s_c:   
    #                 if f_c_tv.get((c,word),0) == 0:
    #                     f_c_tv[c,word] = float(s_c[c,doc,word])
    #                 else:
    #                     f_c_tv[c,word] += float(s_c[c,doc,word])
    #### FIN Caso especial    
    #2º.Para cada clase calculamos P(c|tv) y P(c_not|tv):
    p_c = {}
    for c in clases: 
        p_c[c] = m_c[c]/(len(m)-1) 
        print(f'{c} {m_c[c]} {len(m)-1} - {p_c[c]}')
        for word in v:
            if (c,word) in f_c_tv: 
                p_c_tv[c,word] = f_c_tv[c,word] / f_tv[word]

                if len(m) != f_tv[word]:   
                    p_c_notv[c,word] = (m_c[c] - f_c_tv[c,word]) / (len(m) - f_tv[word])
                else:
                    p_c_notv[c,word] = m_c[c]/(len(m))
            else:
                divisor = max(0.00000001, (len(m)-f_tv[word]))
                p_c_notv[c,word] = m_c[c]/divisor
            p_c_notv[c,word] = max(0.0, p_c_notv[c,word])
    
    print('CALCULANDO EL INFORMATION GAIN DE CADA PALABRA')
    ig = {} #IG de un palabra
    for word in lw_all_total:
        ig[word] = 0
        r1 = 0
        r2 = 0
        r3 = 0
        for c in clases:
            if not p_c[c]:
                p_c[c] += 0.000000000000000000000000001
            r1 += p_c[c]*math.log(p_c[c])
            if (c,word) in p_c_tv and p_c_tv[c,word] != 0.0:
                r2 += p_c_tv[c,word]*math.log(p_c_tv[c,word])
            else:
                r2 += 0.0
            if (c,word) in p_c_notv and p_c_notv[c,word] != 0.0:
                r3 += p_c_notv[c,word]*math.log(p_c_notv[c,word])
            else:
                r3 += 0.0
        ig[word] += -r1 + (p_tv[word] * r2) + (p_notv[word]*r3)
    
    print("ORDENANDO EL DICCIONARIO")
    #Ordenamos el IG de mayor a menor:
    infGain_sort = sorted(ig.items(), key = operator.itemgetter(1), reverse=True)
    i = 0
    print("SACANDO LOS RESULTADOS AL FICHERO EXTERNO")
    with open(args.path_res, 'w') as f:
        for word in infGain_sort:
            if i <= 32768:
                w0 = str(word[0])
                w1 = str(word[1])
                s = str(w0 + ' ' + w1 + '\n')
                f.write(s)
            else: break
            i += 1