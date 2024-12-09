for seed in 0
do
    echo "Running seed $seed"
    python forecast_custom.py \
        --seed $seed \
        --output_path output/forecast_btc \
        --exp_name normal_350_1208 \
        --data_path dataset/btc.csv \
        --index_col timeOpen \
        --target_cols close \
        --train_start_year 2019 \
        --test_start_year 2023 \
        --prediction_length 30 \
        --context_length 30 \
        --epochs 350 \
        --num_batches 64 \
        --num_parallel_samples 1000 \
        --is_pre_scaling 1 \
        --is_model_scaling 0 \
        --add_time_features 0 \
        --add_extention_features 0
done