# HGALayoutLM

## Installation

```
git clone https://github.com/Line-Kite/HGALayoutLM
cd HGALayoutLM
conda create -n hga python=3.7
conda activate hga
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -r requirements.txt
```


## Checkpoints

**Password: 2024**

### Pre-trained Models



| Model               | Model Name (Path)                                                                                              | 
|---------------------|----------------------------------------------------------------------------------------------------------------|
| hgalayoutlm-base  | [hgalayoutlm-base](https://pan.baidu.com/s/1UgvS83sdmyiCbhh1mfIDlQ)  |
| hgalayoutlm-large | [hgalayoutlm-large](https://pan.baidu.com/s/10Cln5iNXCvWkInIZvOP8-g) |


## Finetuning Examples

### CORD

  |Model on CORD                                                                                                                | precision | recall |    f1    | accuracy |
  |:---------------------------------------------------------------------------------------------------------------------------:|:---------:|:------:|:--------:|:--------:|
  | [hgalayoutlm-base-finetuned-cord](https://pan.baidu.com/s/1cMN8urfvHwceZXorMWHbLA)  |   0.9767  | 0.9738 |  0.9753  |  0.9737  |
  | [hgalayoutlm-large-finetuned-cord](https://pan.baidu.com/s/1PtE3Y12_5-Ap-cPUGvFwZg) |   0.9834  | 0.9768 |  0.9801  |  0.9805  |

Download the model weights and move it to the directory "pretrained".

Download the [CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2) dataset and move it to the directory "datasets".

#### Base

```
cd examples
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 20655 run_cord.py \
    --dataset_name cord \
    --do_train \
    --do_eval \
    --model_name_or_path ../pretrained/hgalayoutlm-base  \
    --output_dir ../path/cord/base/test \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --max_steps 2000 --save_steps -1 --evaluation_strategy steps --eval_steps 100 \
    --learning_rate 5e-5 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 --overwrite_output_dir
```

#### Large

```
cd examples
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 20655 run_cord.py \
    --dataset_name cord \
    --do_train \
    --do_eval \
    --model_name_or_path ../pretrained/hgalayoutlm-large  \
    --output_dir ../path/cord/large/test \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --max_steps 2000 --save_steps -1 --evaluation_strategy steps --eval_steps 100 \
    --learning_rate 5e-5 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 --overwrite_output_dir
```


## Citation
Please cite our paper if the work helps you.
```
@inproceedings{li2024hypergraph,
  title={Hypergraph based Understanding for Document Semantic Entity Recognition},
  author={Li, Qiwei and Li, Zuchao and Wang, Ping and Ai, Haojun and Zhao, Hai},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={2950--2960},
  year={2024}
}

```


## Note

We will follow-up complement other examples.
