for seed in 0
do
    echo "Running seed $seed"
    python forecast_custom.py \
        --seed $seed \
        --output_path output/forecast_btc \
        --exp_name point_forecast_1210 \
        --data_path dataset/btc.csv \
        --index_col timeOpen \
        --target_cols close \
        --test_start_date "2023-08-01" \
        --train_data_length 60 \
        --test_data_length 30 \
        --split_type index \
        --prediction_length 30 \
        --context_length 30 \
        --epochs 300 \
        --num_batches 64 \
        --num_parallel_samples 1000 \
        --is_pre_scaling 1 \
        --is_model_scaling 0 \
        --add_time_features 0 \
        --add_extention_features 0
done