for seed in 0
do
    for start_date in "2023-01-02"
    do
        echo "Running seed $seed"
        python forecast_custom.py \
            --seed $seed \
            --output_path output/forecast_jpy \
            --exp_name point_forecast_1210 \
            --data_path dataset/usd_jpy.csv \
            --index_col Date \
            --target_cols Price \
            --test_start_date ${start_date} \
            --train_data_length 1000 \
            --test_data_length 360 \
            --val_data_length 180 \
            --split_type index \
            --prediction_length 30 \
            --context_length 30 \
            --epochs 500 \
            --num_batches 64 \
            --num_parallel_samples 1000 \
            --is_pre_scaling 1 \
            --is_model_scaling 1 \
            --add_time_features 0 \
            --add_extention_features 0
    done
done