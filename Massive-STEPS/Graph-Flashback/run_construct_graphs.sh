for city in bandung beijing istanbul jakarta kuwait_city melbourne moscow new_york palembang petaling_jaya sao_paulo shanghai sydney tangerang tokyo; do
    python KGE/construct_loc_loc_graph.py \
        --model_type transe \
        --dataset $city \
        --pretrain_model ../KGE-Graph-Flashback/log/$city-transe.ckpt \
        --version scheme2 \
        --dataset_train_file data/${city}/${city}_checkins_train.csv \
        --dataset_val_file data/${city}/${city}_checkins_validation.csv \
        --dataset_test_file data/${city}/${city}_checkins_test.csv

    python KGE/construct_user_loc_graph.py \
        --model_type transe \
        --dataset $city \
        --pretrain_model ../KGE-Graph-Flashback/log/$city-transe.ckpt \
        --version scheme2 \
        --dataset_train_file data/${city}/${city}_checkins_train.csv \
        --dataset_val_file data/${city}/${city}_checkins_validation.csv \
        --dataset_test_file data/${city}/${city}_checkins_test.csv
done