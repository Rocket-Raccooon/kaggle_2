python train.py --model_name resnet50
python train.py --model_name resnet152
python train.py --model_name efficientnet_b3
python train.py --model_name efficientnet_b4
python train.py --model_name regnet_x_8gf

python submit_maker.py --model_log weights_resnet50/logs.txt
python submit_maker.py --model_log weights_resnet152/logs.txt
python submit_maker.py --model_log weights_efficientnet_b3/logs.txt
python submit_maker.py --model_log weights_efficientnet_b4/logs.txt
python submit_maker.py --model_log weights_regnet_x_8gf/logs.txt

python predicts_averaging.py --source "weights_regnet_x_8gf_scores.csv;weights_efficientnet_b4_scores.csv;weights_efficientnet_b3_scores.csv;weights_resnet152_scores.csv;weights_resnet152_scores.csv" --out_name resnets__effbs_regnet_weighted