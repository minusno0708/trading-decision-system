for seed in 0
do
    #for start_date in "2023-06-01" "2023-08-01" "2023-10-01" "2023-12-01" "2024-02-01" "2024-04-01" "2024-06-01"
    #for start_date in "2023-06-01" "2023-07-01" "2023-08-01" "2023-09-01" "2023-10-01" "2023-11-01" "2023-12-01" "2024-01-01" "2024-02-01" "2024-03-01" "2024-04-01" "2024-05-01" "2024-06-01"
    for start_date in "2023-01-01"
    do
        echo "Running seed $seed"
        python forecast_custom.py \
            --seed $seed \
            --output_path output/forecast_btc \
            --exp_name point_forecast_1214 \
            --data_path dataset/btc.csv \
            --index_col timeOpen \
            --target_cols close \
            --test_start_date ${start_date} \
            --train_data_length 1000 \
            --test_data_length 360 \
            --val_data_length 180 \
            --split_type index \
            --prediction_length 1 \
            --context_length 30 \
            --epochs 300 \
            --num_batches 64 \
            --num_parallel_samples 1000 \
            --is_pre_scaling 1 \
            --is_model_scaling 1 \
            --add_time_features 0 \
            --add_extention_features 0
    done
done