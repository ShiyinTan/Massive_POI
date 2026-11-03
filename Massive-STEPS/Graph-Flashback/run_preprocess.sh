for city in bandung beijing istanbul jakarta kuwait_city melbourne moscow new_york palembang petaling_jaya sao_paulo shanghai sydney tangerang tokyo; do
    python KGE/generate_triplet.py \
        --dataset_train_file data/${city}/${city}_checkins_train.csv \
        --dataset_val_file data/${city}/${city}_checkins_validation.csv \
        --dataset_test_file data/${city}/${city}_checkins_test.csv \
        --city $city

    python KGE/refine.py --city $city
done