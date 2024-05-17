





# pretrain
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234  train_finetune_common_cls_RCT.py --config iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_noBN_noRGBweights | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1232  train_finetune_common_cls_RCT.py --config iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_wBN_noRGBweights | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1235  train_finetune_common_cls_RCT.py --config iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_noBN_wRGBweights | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.4" --master_port=1234  train_finetune_common_cls_RCT.py --config iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_wBN_wRGBweights | tee hist.log


cd ../evaluate

# eval
python eval_lamp_hq.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_noBN_noRGBweight' --architecture iresnet18 --gpu_ids 01234567  --fold 1
python eval_casia.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_noBN_noRGBweight' --architecture iresnet18 --gpu_ids 01234567  --fold 1
python eval_oulu_casia.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_noBN_noRGBweight' --architecture iresnet18 --gpu_ids 01234567
python eval_buua.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_noBN_noRGBweight' --architecture iresnet18 --gpu_ids 01234567

python eval_lamp_hq.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_wBN_noRGBweight' --architecture iresnet18 --gpu_ids 01234567  --fold 1
python eval_casia.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_wBN_noRGBweight' --architecture iresnet18 --gpu_ids 01234567  --fold 1
python eval_oulu_casia.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_wBN_noRGBweight' --architecture iresnet18 --gpu_ids 01234567
python eval_buua.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_wBN_noRGBweight' --architecture iresnet18 --gpu_ids 01234567

python eval_lamp_hq.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_noBN_wRGBweight' --architecture iresnet18 --gpu_ids 01234567  --fold 1
python eval_casia.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_noBN_wRGBweight' --architecture iresnet18 --gpu_ids 01234567  --fold 1
python eval_oulu_casia.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_noBN_wRGBweight' --architecture iresnet18 --gpu_ids 01234567
python eval_buua.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_noBN_wRGBweight' --architecture iresnet18 --gpu_ids 01234567

python eval_lamp_hq.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_wBN_wRGBweight' --architecture iresnet18 --gpu_ids 01234567  --fold 1
python eval_casia.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_wBN_wRGBweight' --architecture iresnet18 --gpu_ids 01234567  --fold 1
python eval_oulu_casia.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_wBN_wRGBweight' --architecture iresnet18 --gpu_ids 01234567
python eval_buua.py --checkpoint '/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT_wBN_wRGBweight' --architecture iresnet18 --gpu_ids 01234567

