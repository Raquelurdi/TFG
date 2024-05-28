cd ..
path_resultados=work_JMBD4949_4950_loo_groups_other
mkdir ${path_resultados}
#### Information Gain
data_path=PrIx/JMBD4949_4950_idx
prob=0.1
classes=P,CP,O,A,T,V,R,CEN,DP,D,C,TH
path_res_IG=${path_resultados}/resultadosIG_4949_4950.txt
python infogain_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res ${path_res_IG} --all_files True
#### TFIDF
path_res_train=${path_resultados}/tfidf_4949_4950_loo.txt
path_res_classes=${path_resultados}/tfidf_4949_4950_loo_classes.txt
python tf_idf_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res_train ${path_res_train} --IG_file ${path_res_IG} --path_res_classes ${path_res_classes} --all_files True
cd launchers