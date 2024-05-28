#!/usr/bin/env bash
cd ..
trys=( 1 )
ngfs=( 128,128 )
#ngfs=( 16 32,16 128,64,32  )
numfeats=$(seq 11 11 | xargs -n 1 -I {} echo "2^"{} | bc)
# numfeats=$(seq 2 2 | xargs -n 1 -I {} echo "2^"{} | bc)
#res_10kwords res_10kwords_withoutNormalize res_10kwords_normRWs
classes=P,CP,O,A,T,V,R,CEN,DP,D,C,TH
for try in $trys; do
    for ngf in "${ngfs[@]}"; do
#        for numfeat in "${numfeats[@]}"; do
        for numfeat in $numfeats; do
            python main.py --epochs 50 --work_dir works_JMBD4949_JMBD4950_prod_12classes/work_${ngf}_numFeat${numfeat} \
            --layers ${ngf} --batch_size 64 --lr 0.01 --optim ADAM \
            --num_workers 0 --seed ${try}  --num_feats ${numfeat} \
            --tr_data work_JMBD4949_JMBD4950_prod/tfidf_4949_4950_prod_tr.txt \
            --prod_data work_JMBD4949_JMBD4950_prod/tfidf_4949_4950_prod_prod.txt --do_prod true --do_test false \
            --class_dict work_JMBD4949_JMBD4950_prod/tfidf_4949_4950_classes.txt --LOO false --classes ${classes} --auto_lr_find true
        done
    done
done
cd launchers