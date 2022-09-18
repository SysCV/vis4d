#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=60G
#SBATCH  --constraint='geforce_rtx_2080_ti|titan_xp|geforce_gtx_titan_x|titan_x|geforce_gtx_1080_ti'

python -m vis4d.op.segment.fcn_voc_training --resnet_model "resnet50" --save_name "fcn_resnet50_voc2012"