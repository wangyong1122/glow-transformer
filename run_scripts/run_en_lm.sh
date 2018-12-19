CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=44456 \
                ez_run.py \
                --prefix [time] \
                --mode train \
                --data_prefix "./data/" \
                --dataset "wmt16" \
                --src "en" --trg "ro" \
                --test_src "en" --test_trg "ro" \
                --train_set "train" \
                --dev_set   "dev"   \
                --test_set  "test"  \
                --load_lazy \
                --workspace_prefix "/data1/Glow-lm/" \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size 1200 \
                --inter_size 4 \
                --label_smooth 0.1 \
                --lr 0.0005 \
                --weight_decay 0.0001 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "last_layer" \
                --model 'AutoTransformer2' \
               # --debug --no_valid