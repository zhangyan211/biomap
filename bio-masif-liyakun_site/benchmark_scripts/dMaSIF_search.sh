# Load environment

python -W ignore main_training.py \
    --experiment_name=dMaSIF_search_1layer_12A \
    --batch_size=64 \
    --embedding_layer=dMaSIF \
    --search=True \
    --device=cuda:0 \
    --random_rotation=True \
    --radius=12.0 \
    --n_layers=1 \
    --debug=1

# python -W ignore main_training.py \
#     --experiment_name dMaSIF_search_3layer_12A \
#     --batch_size 64 \
#     --embedding_layer dMaSIF \
#     --search True \
#     --device cuda:0 \
#     --random_rotation True \
#     --radius 12.0 \
#     --n_layers 3

