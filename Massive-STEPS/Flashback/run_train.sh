for city in bandung istanbul jakarta kuwait_city melbourne moscow new_york palembang petaling_jaya shanghai sao_paulo sydney tangerang tokyo; do
    python train.py \
        --dataset_train_file data/$city/${city}_checkins_train.csv \
        --dataset_val_file data/$city/${city}_checkins_validation.csv \
        --dataset_test_file data/$city/${city}_checkins_test.csv \
        --city $city \
        --batch-size 100
done

for city in beijing; do
    python train.py \
        --dataset_train_file data/$city/${city}_checkins_train.csv \
        --dataset_val_file data/$city/${city}_checkins_validation.csv \
        --dataset_test_file data/$city/${city}_checkins_test.csv \
        --city $city \
        --batch-size 10
done