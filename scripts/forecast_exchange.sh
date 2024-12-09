for seed in 0
do
    echo "Running seed $seed"
    python forecast_custom.py \
        --seed $seed \
        --output_path output/forecast_jpy \
        --exp_name normal_350_1208 \
        --data_path dataset/usd_jpy.csv \
        --index_col Date \
        --target_cols Price \
        --train_start_date "1972-01-01" \
        --train_end_date "2022-12-31" \
        --test_start_date "2023-01-01" \
        --test_end_date "2023-12-31" \
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