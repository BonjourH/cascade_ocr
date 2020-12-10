i=1
for lr in 0.001 0.0005 0.0001
do
    for wd in 0.005 0.001
    do
        python -u train.py \
                    --lr $lr \
                    --weight_decay $wd \
                    --Exp_ID $i \
                    --device_id 2 \
                    --epochs 300 \
                    --patience 10
        let i++
    done
done

