gpus=${1:-2}
jbname=${2:-Insertable_JA}
mode=${3:-train}
load_from=${4:-none}  # --load_from name --resume
python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port=23456 \
                ez_run.py \
                --prefix l2r \
                --mode ${mode} \
                --data_prefix "/private/home/jgu/data/" \
                --dataset "kftt" \
                --src "en" --trg "ja" \
                --train_set "train.sub.shuf.l2r" \
                --dev_set   "dev.sub"  \
                --vocab_file "en-ja/vocab.en-ja.n.ins.pt" \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/private/home/jgu/space/${jbname}/" \
                --eval_every 500  \
                --batch_size 2000 \
                --sub_inter_size 1 \
                --inter_size 8 \
                --label_smooth 0.1 \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --load_from ${load_from} \
                --length_ratio 2 \
                --beam_size 10 \
                --relative_pos \
                --model TransformerIns \
                --insertable --insert_mode word_first \
                --order fixed \
                # --debug --no_valid
                #--debug --no_valid
                
