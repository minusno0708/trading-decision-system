for seed in 0
do
    for start_date in "2020-11-06"
    do
        echo "Running seed $seed"
        python forecast_custom.py \
            --seed $seed \
            --output_path output/forecast_wth \
            --exp_name point_forecast_1214 \
            --data_path dataset/weather.csv \
            --index_col date \
            --target_cols "T (degC)" \
            --test_start_date ${start_date} \
            --train_data_length 36600 \
            --test_data_length 7904 \
            --val_data_length 7905 \
            --split_type index \
            --prediction_length 1 \
            --context_length 30 \
            --epochs 1 \
            --num_batches 64 \
            --num_parallel_samples 1000 \
            --is_pre_scaling 1 \
            --is_model_scaling 1 \
            --add_time_features 0 \
            --add_extention_features 0
    done
done