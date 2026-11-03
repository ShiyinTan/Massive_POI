for city in bandung beijing istanbul jakarta kuwait_city melbourne moscow new_york palembang petaling_jaya sao_paulo shanghai sydney tangerang tokyo; do
    python convert_std.py \
        --city $city \
        --path data \
        --output_path raw_data

    for model in FPMC RNN DeepMove; do
        python run_model.py \
            --task traj_loc_pred \
            --model $model \
            --dataset std_$city \
            --config libcity/config/data/STDDataset
    done

    for model in LSTPM; do
        python run_model.py \
            --task traj_loc_pred \
            --model $model \
            --dataset std_$city \
            --config libcity/config/data/STDDataset$model
    done
done