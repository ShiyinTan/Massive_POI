for city in bandung beijing istanbul jakarta kuwait_city melbourne moscow new_york palembang petaling_jaya sao_paulo shanghai sydney tangerang tokyo; do
    python build_graph.py \
        --csv_path data/$city/${city}_checkins_train.csv \
        --output_dir data/$city
done

for city in beijing melbourne moscow palembang shanghai sydney tokyo; do
    python train.py \
        --data-train data/$city/${city}_checkins_train.csv \
        --data-val data/$city/${city}_checkins_test.csv \
        --data-adj-mtx data/$city/graph_A.csv \
        --data-node-feats data/$city/graph_X.csv \
        --time-units 48 --timestamp_column timestamp \
        --poi-embed-dim 128 --user-embed-dim 128 \
        --time-embed-dim 32 --cat-embed-dim 32 \
        --node-attn-nhid 128 \
        --transformer-nhid 1024 \
        --transformer-nlayers 2 --transformer-nhead 2 \
        --batch 16 --epochs 200 --name $city \
        --workers 12 --exist-ok \
        --lr 0.001
done

for city in bandung istanbul kuwait_city new_york petaling_jaya sao_paulo tangerang; do
    python train.py \
        --data-train data/$city/${city}_checkins_train.csv \
        --data-val data/$city/${city}_checkins_test.csv \
        --data-adj-mtx data/$city/graph_A.csv \
        --data-node-feats data/$city/graph_X.csv \
        --time-units 48 --timestamp_column timestamp \
        --poi-embed-dim 128 --user-embed-dim 128 \
        --time-embed-dim 32 --cat-embed-dim 32 \
        --node-attn-nhid 128 \
        --transformer-nhid 1024 \
        --transformer-nlayers 2 --transformer-nhead 2 \
        --batch 16 --epochs 20 --name $city \
        --workers 12 --exist-ok \
        --lr 0.0001
done

for city in jakarta; do
    python train.py \
        --data-train data/$city/${city}_checkins_train.csv \
        --data-val data/$city/${city}_checkins_test.csv \
        --data-adj-mtx data/$city/graph_A.csv \
        --data-node-feats data/$city/graph_X.csv \
        --time-units 48 --timestamp_column timestamp \
        --poi-embed-dim 128 --user-embed-dim 128 \
        --time-embed-dim 32 --cat-embed-dim 32 \
        --node-attn-nhid 128 \
        --transformer-nhid 1024 \
        --transformer-nlayers 2 --transformer-nhead 2 \
        --batch 16 --epochs 20 --name $city \
        --workers 12 --exist-ok \
        --lr 0.00005
done