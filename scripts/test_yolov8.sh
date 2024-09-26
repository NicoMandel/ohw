#!/usr/bin/bash
models1=("20240924-n-1cm-1cm" "20240924-s-1cm-1cm" "20240924-m-1cm-1cm" "20240925-l-1cm-1cm") 
save_file="results/yolov8.xlsx"

ds="024cm"
for md in "${models1[@]}"
do
    python "scripts/test_model.py" "results/$md/weights/best.pt" $ds -s $save_file
done

models2=("20240924-n-024cm-024cm" "20240924-s-024cm-024cm" "20240925-m-024cm-024cm" "20240925-l-024cm-024cm")
ds="1cm"
for md in "${models2[@]}"
do
    python "scripts/test_model.py" "results/$md/weights/best.pt" $ds -s $save_file
done