#!/usr/bin/env bash
cd ..
trys=( 1 )
# ngfs=( 128,128 )
ngfs=( 128,128 )
classes=P,CP,O,A,T,V,R,CEN,DP,D,C,TH
#ngfs=( 16 32,16 128,64,32  )
# numfeats=$(seq 4 15 | xargs -n 1 -I {} echo "2^"{} | bc)
#ngfs=( 0 )
numfeats=$(seq 11 11 | xargs -n 1 -I {} echo "2^"{} | bc)
path=work_JMBD4949_4950_loo_groups_trans_12classes_prod
for try in $trys; do
    for ngf in "${ngfs[@]}"; do
        for numfeat in $numfeats; do
            python main.py --epochs 500 --work_dir works_JMBD4949_4950_loo_groups_trans_12clases_prod/work_${ngf}_numFeat${numfeat} \
            --layers ${ngf} --batch_size 50 --lr 0.01 --optim ADAM \
            --num_workers 0 --seed ${try}  --num_feats ${numfeat} \
            --tr_data ${path}/tfidf_4949_4950_prod_tr.txt \
            --prod_data ${path}/tfidf_4949_4950_prod_prod.txt --do_prod true --do_test false \
            --class_dict ${path}/tfidf_4949_4950_classes.txt \
            --classes ${classes} --LOO false --auto_lr_find true
        done
    done
done
cd launchers
