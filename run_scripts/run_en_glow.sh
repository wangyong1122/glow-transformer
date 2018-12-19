CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=54456 \
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
                --load_from "12.17_15.50.26..wmt16_t2t-base_en_ro_wf_bpe_0.1_4800_" \
                --params "t2t-base" \
                --eval_every 200  \
                --batch_size 2400 \
                --inter_size 4 \
                --label_smooth 0.1 \
                --lr 0.0005 \
                --lr_glow 0.0001 \
                --weight_decay 0.0001 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "last_layer" \
                --model 'AutoTransformer2' \
                --glow_mode \
               # --debug --no_valid