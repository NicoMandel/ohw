#!/usr/bin/bash
models=("n" "s" "m" "l") #
dss=("1cm" "024cm")

for md in "${models[@]}"
do
    for ds in "${dss[@]}"
    do
        python "scripts/train_model.py" $md $ds -s "results/yolov8.xlsx"
    done
done