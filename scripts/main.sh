#!/bin/bash

# python run.py --exp_name easyscene --data_dir ./data --out_dir ./results

base_folder="no_chessboard"
for folder in data/no_chessboard/*; do
    if [[ -d "$folder" ]]; then
        expname=$(basename "$folder")
        # Use the expname variable for further processing
        echo "Processing $expname"
        main_folder="$base_folder/$expname"
        python run.py --exp_name "motion" --data_dir "data/$main_folder" --out_dir "./results/$main_folder" --num_imgs 8
        python run.py --exp_name "motion" --data_dir "data/$main_folder" --out_dir "./results/$main_folder" --num_imgs 10
        python run.py --exp_name "motion" --data_dir "data/$main_folder" --out_dir "./results/$main_folder" --num_imgs 12
        python run.py --exp_name "noise_0.00" --data_dir "data/$main_folder" --out_dir "./results/$main_folder"
        python run.py --exp_name "noise_0.50" --data_dir "data/$main_folder" --out_dir "./results/$main_folder"
        python run.py --exp_name "noise_1.00" --data_dir "data/$main_folder" --out_dir "./results/$main_folder"
        python run.py --exp_name "noise_1.50" --data_dir "data/$main_folder" --out_dir "./results/$main_folder"

    fi
done

# base_dir="results/no_chessboard/no_noise"
# start_index=40
# counter=$start_index
# for folder in "$base_dir"/*; do
#   if [ -d "$folder" ]; then
#     # idx=$((counter + 10))
#     new_name=$(printf "%03d_10" "$counter"_10)
#     mv "$folder" "$base_dir/$new_name"
#     echo "Renamed $folder to $base_dir/$new_name"
#     ((counter++))
#   fi
# done
# echo "Renaming completed."

