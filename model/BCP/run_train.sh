
IMG_PATH="RiskBench_Dataset/"
BS_PATH="RiskBench_Dataset/bs_data"     # bev-seg data
PF_PATH="RiskBench_Dataset/pf_data"     # potential field data
METADATA_PATH="metadata/RiskBench"

# BCP
# python train.py --method vision --data_root $IMG_PATH --meta_root $METADATA_PATH --batch_size 8 --lr 0.0000001 --loss_weights "[1.0,1.6]" --verbose

# TP+BCP
# python train.py --method vision --data_root $IMG_PATH --meta_root $METADATA_PATH --batch_size 8 --lr 0.00001 --loss_weights "[1.0,1.2]" --use_target_point --verbose 

# BS+BCP
# python train.py --method bev_seg --data_root $BS_PATH --meta_root $METADATA_PATH --batch_size 128 --lr 0.00001 --loss_weights "[1.0,1.2]" --use_gt --verbose
# python train.py --method bev_seg --data_root $BS_PATH --meta_root $METADATA_PATH --batch_size 128 --lr 0.000001 --loss_weights "[1.0,1.2]" --use_gt --use_target_point --verbose

# PF+BCP
# python train.py --method pf --data_root $PF_PATH --meta_root $METADATA_PATH --batch_size 128 --lr 0.000001 --loss_weights "[1.0,1.2]" --use_gt --verbose
