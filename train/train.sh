#export OMP_NUM_THREADS=4
#
#starttime=`date +'%Y-%m-%d %H:%M:%S'`
#
#
## /usr/bin/
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581  pretrain.py | tee hist.log
#
#
#endtime=`date +'%Y-%m-%d %H:%M:%S'`
#start_seconds=$(date --date="$starttime" +%s)
#end_seconds=$(date --date="$endtime" +%s)
#echo "运行时间:"$((end_seconds-start_seconds))"s"
#
#ps -ef | grep "partial_fc" | grep -v grep | awk '{print "kill -9 "$2}' | sh

#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234  pretrain.py --config efficientnet_b0 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234  pretrain.py --config efficientnet_b0_rand-red | tee hist.log

# Augmented RGB training --------------------------------------------------------------
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  pretrain.py --config config_iresnet100_ReCA | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234  pretrain.py --config config_iresnet100_RaCA | tee hist.log

#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234  pretrain.py --config config_iresnet50_RaCA | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  pretrain.py --config config_iresnet50_ReCA | tee hist.log

#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_aug_dataloader.py --config config_iresnet18_ms1m_small_aug_a001_b001  | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_aug_dataloader.py --config config_iresnet18_ms1m_small_aug_a01_b005  | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_aug_dataloader.py --config config_iresnet9_ms1m_small_aug_a01_b005  | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_aug_dataloader.py --config config_iresnet9_ms1m_small_aug_a01_b01  | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_aug_dataloader.py --config config_iresnet9_ms1m_small_aug_a005_b002  | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_aug_dataloader.py --config config_iresnet18_ms1m_small_aug_a01_b01  | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_aug_dataloader.py --config config_iresnet18_ms1m_small_aug_a005_b002  | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_aug_dataloader.py --config config_iresnet9_ms1m_small_aug_a001_b001  | tee hist.log
#
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  pretrain.py --config config_iresnet100_RaCA  | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  pretrain.py --config config_mobilefacenet  | tee hist.log
#

#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_S_MobileNet_MS1M_MS1M | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_S_MobileNet_MS1M_MS1M | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd2_aug.py --config T_IR50_S_MobileNet_MS1Maug_MS1Maug | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd2_finetune_NIR.py --config T_IR50_S_MobileNet_MS1M_LampHQ | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config mobilenet_init-MS1M_LAMP-HQ1of10_RCT | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd2_finetune_modalityCenters.py --config T_IR50_S_MobileNet_MS1M_LampHQ_modNone | tee hist.log
#
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls_RCT_cls_then_backbone.py --config mobilenet_init-MS1M_LAMP-HQ1of10_RCT_2stage_classif_then_backbone  | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config iresnet50_init-MS1M_LAMP-HQ1of10_RCT | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_LampHQ_S_MobileNet_MS1M | tee hist.log
#
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_LampHQ_S_MobileNet_MS1M_noCLSLoss | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config iresnet50_init-MS1M_LAMP-HQ1of10_RCT | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls_RCT_cls_then_backbone.py --config mobilenet_init-MS1M_LAMP-HQ1of10_RCT_2stage_classif_then_backbone | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_S_MobileNet_MS1M_MS1M | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_LampHQ_S_MobileNet_MS1M | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config mobilenet_init-MS1M_LAMP-HQ1of10_RCT | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_S_MobileNet_MS1M_MS1M | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls_RCT_wUnetInput.py --config mobilenet_init-MS1M_LAMP-HQ1of10_RCT | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls_RCT_wUnetInput2.py --config mobilenet_init-MS1M_LAMP-HQ1of10_RCT_wTeacher | tee hist.log
#
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_ekd.py --config T_IR50_ReCA_S_MobileNet_CLS_MS1M_EKD_MS1M | tee hist.log

#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config iresnet50_init-MS1M_CASIA1of10_RCT | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config iresnet50_init-MS1M_OULU-CASIA_RCT | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config iresnet50_init-MS1M_BUAA_RCT | tee hist.log

#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config mobilenet_init-MS1M_LAMP-HQ1of10_RCT | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config mobilenet_init-MS1M_CASIA1of10_RCT | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config mobilenet_init-MS1M_OULU-CASIA_RCT | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config mobilenet_init-MS1M_BUAA_RCT | tee hist.log

#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_CASIA_S_MobileNet_MS1M | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_OuluCASIA_S_MobileNet_MS1M | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_BUAA_S_MobileNet_MS1M | tee hist.log

#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config mobilenet_init-MS1M_CASIA1of10_RCT_S1 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config mobilenet_init-MS1M_BUAA_RCT_S1 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config mobilenet_init-MS1M_BUAA_RCT | tee hist.log

#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_BUAA_S_MobileNet_MS1M | tee hist.log

#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_CASIA_S_MobileNet_MS1M | tee hist.log

#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config mobilenet_init-MS1M_BUAA_RCT_S1 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config mobilenet_init-MS1M_CASIA1of10_RCT_S1 | tee hist.log

#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config mobilenet_init-MS1M_OULU-CASIA_RCT_S1 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  finetune_MEC.py --config mobilenet_init-MS1M_OULU-CASIA_RCT_S1 | tee hist.log

#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_common_cls.py --config iresnet50_init-MS1M_LAMP-HQ1of10_common-cls_bn | tee hist.log


#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  finetune_MEC.py --config lightcnn29_init-MS1M_CASIA1of10_RCT | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  finetune_MEC.py --config lightcnn29_init-MS1M_OULU-CASIA_RCT | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  finetune_MEC.py --config lightcnn29_init-MS1M_BUAA_RCT | tee hist.log
#
#

# 92
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config mobilenet_init-MS1M-ReCA_LampHQ1of10 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config mobilenet_init-rand_BUAA | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config mobilenet_init-rand_LampHQ1of10 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config mobilenet_init-MS1M-ReCA_BUAA | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config mobilenet_init-MS1M-ReCA_CASIA1of10 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config mobilenet_init-MS1M-ReCA_OuluCASIA | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config mobilenet_init-rand_OuluCASIA | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config mobilenet_init-rand_CASIA1of10 | tee hist.log
##################################
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config mobilenet_init-MS1M-ReCA_LampHQ1of10_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config mobilenet_init-MS1M-ReCA_BUAA_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config mobilenet_init-MS1M-ReCA_CASIA1of10_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config mobilenet_init-MS1M-ReCA_OuluCASIA_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet100_init-MS1M-ReCA_BUAA | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet100_init-MS1M-ReCA_BUAA_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet100_init-MS1M-ReCA_CASIA1of10 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet100_init-MS1M-ReCA_CASIA1of10_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet100_init-MS1M-ReCA_LampHQ1of10 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet100_init-MS1M-ReCA_LampHQ1of10_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet100_init-MS1M-ReCA_OuluCASIA | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet100_init-MS1M-ReCA_OuluCASIA_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  finetune_MEC.py --config iresnet18_init-MS1M-ReCA_LampHQ1of10_no-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  finetune_MEC.py --config iresnet18_init-MS1M-ReCA_OuluCASIA_no-reg | tee hist.log
##python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  finetune_MEC.py --config iresnet50_init-MS1M-ReCA_BUAA_no-reg | tee hist.log
##python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  finetune_MEC.py --config iresnet50_init-MS1M-ReCA_CASIA1of10_no-reg | tee hist.log
##python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  finetune_MEC.py --config iresnet50_init-MS1M-ReCA_LampHQ1of10_no-reg | tee hist.log
##python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  finetune_MEC.py --config iresnet50_init-MS1M-ReCA_OuluCASIA_no-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  finetune_MEC.py --config iresnet18_init-MS1M-ReCA_CASIA1of10_no-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  finetune_MEC.py --config iresnet18_init-MS1M-ReCA_BUAA_no-reg | tee hist.log


python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1234  train_dataloader_simkd.py --config T_IR50_S_MobileNet_MS1M_MS1M | tee hist.log




# 76
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet18_init-MS1M-ReCA_BUAA | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet18_init-MS1M-ReCA_CASIA1of10 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet18_init-MS1M-ReCA_LampHQ1of10 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet18_init-MS1M-ReCA_OuluCASIA | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet18_init-rand_BUAA | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet18_init-rand_CASIA1of10 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet18_init-rand_LampHQ1of10 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet18_init-rand_OuluCASIA | tee hist.log
################################################################## HAVENT SETUP YET - NEED TO RUN
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet18_init-MS1M-ReCA_BUAA_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet18_init-MS1M-ReCA_CASIA1of10_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet18_init-MS1M-ReCA_LampHQ1of10_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet18_init-MS1M-ReCA_OuluCASIA_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet50_init-MS1M-ReCA_BUAA | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet50_init-MS1M-ReCA_BUAA_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet50_init-MS1M-ReCA_CASIA1of10 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet50_init-MS1M-ReCA_CASIA1of10_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet50_init-MS1M-ReCA_LampHQ1of10 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet50_init-MS1M-ReCA_LampHQ1of10_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet50_init-MS1M-ReCA_OuluCASIA | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_finetune_cls.py --config iresnet50_init-MS1M-ReCA_OuluCASIA_nBN | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config iresnet18_init-MS1M-ReCA_LampHQ1of10_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config iresnet18_init-MS1M-ReCA_OuluCASIA_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config iresnet18_init-MS1M-ReCA_CASIA1of10_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config iresnet18_init-MS1M-ReCA_BUAA_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config iresnet50_init-MS1M-ReCA_BUAA_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config iresnet50_init-MS1M-ReCA_CASIA1of10_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config iresnet50_init-MS1M-ReCA_LampHQ1of10_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config iresnet50_init-MS1M-ReCA_OuluCASIA_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config iresnet100_init-MS1M-ReCA_OuluCASIA_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config iresnet100_init-MS1M-ReCA_LampHQ1of10_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config iresnet100_init-MS1M-ReCA_CASIA1of10_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config iresnet100_init-MS1M-ReCA_BUAA_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config mobilefacenet_init-MS1M-ReCA_BUAA_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config mobilefacenet_init-MS1M-ReCA_CASIA1of10_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config mobilefacenet_init-MS1M-ReCA_LampHQ1of10_w-reg | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_withreg.py --config mobilefacenet_init-MS1M-ReCA_OuluCASIA_w-reg | tee hist.log

#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  finetune_MEC.py --config mobilefacenet_init-MS1M-ReCA_LampHQ1of10 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config iresnet100_init-MS1M-ReCA_LampHQ1of10_no-reg | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config iresnet100_init-MS1M-ReCA_OuluCASIA_no-reg | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config iresnet100_init-MS1M-ReCA_CASIA1of10_no-reg | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config iresnet100_init-MS1M-ReCA_BUAA_no-reg | tee hist.log


python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_dataloader_simkd_finetune.py --config T_IR50_MS1M_LampHQ_S_MobileNet_MS1M_LampHQ | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_dataloader_simkd_finetune.py --config T_IR50_MS1M_S_MobileNet_MS1M_LampHQ | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR100_S_MobileNet_MS1M_MS1M | tee hist.log


python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_S_MobileNet_MS1M_MS1M | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_dataloader_simkd-wproj_finetune.py --config T_IR50_MS1M_S_MobileNet_MS1M_LampHQ | tee hist.log



python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_dataloader_simkd_finetune.py --config T_IR100_MS1M_S_MobileNet_MS1M_LampHQ | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_dataloader_simkd_finetune.py --config T_IR100_MS1M_LampHQ_S_MobileNet_MS1M_LampHQ | tee hist.log


python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_dataloader_simkd_finetune.py --config T_IR50_MS1M_BUAA_S_MobileNet_MS1M_BUAA | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_dataloader_simkd_finetune.py --config T_IR50_MS1M_CASIA_S_MobileNet_MS1M_CASIA | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_dataloader_simkd_finetune.py --config T_IR50_MS1M_OuluCasia_S_MobileNet_MS1M_OuluCasia | tee hist.log


python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_OuluCASIA_S_MobileNet_MS1M | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_simkd.py --config T_IR50_CASIA_S_MobileNet_MS1M | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_dataloader_kldiv_finetune.py --config T_IR50_MS1M_LampHQ_S_MobileNet_MS1M_LampHQ | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_kldiv.py --config T_IR50_S_MobileNet_MS1M_MS1M | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_arcface.py --config confg_iresnet50_ReCA | tee hist.log



python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config iresnet18_init-MS1M-ReCA_BUAA_no-reg | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config iresnet18_init-MS1M-ReCA_CASIA1of10_no-reg | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config iresnet18_init-MS1M-ReCA_LampHQ1of10_no-reg | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config iresnet18_init-MS1M-ReCA_OuluCASIA_no-reg | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config lightcnn29_init-MS1M_BUAA_RCT | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config lightcnn29_init-MS1M_CASIA1of10_RCT | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config lightcnn29_init-MS1M_LAMP-HQ1of10_RCT | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config lightcnn29_init-MS1M_OULU-CASIA_RCT | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config mobilefacenet_init-MS1M-ReCA_BUAA | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config mobilefacenet_init-MS1M-ReCA_CASIA1of10 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config mobilefacenet_init-MS1M-ReCA_LampHQ1of10 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config mobilefacenet_init-MS1M-ReCA_OuluCASIA | tee hist.log


python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader_arcface.py --config confg_MobileNet_MS1M | tee hist.log

#also do pretrainings with smaller or larger margin


python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config mobilefacenet_init-MS1M-ReCA_LampHQ1of10 | tee hist.log



python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_dataloader_simkd_finetune.py --config T_IR18_MS1M_S_MobileNet_MS1M_LampHQ | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_dataloader_simkd-wproj_finetune.py --config T_IR50_MS1M_S_MobileNet_MS1M_LampHQ | tee hist.log






#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  finetune_joint_MEC.py --config mobilefacenet_init-MS1M-ReCA_LampHQ1of10 | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  finetune_joint_MEC.py --config mobilefacenet_init-MS1M-ReCA_CASIA1of10 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config iresnet18_init-MS1M-ReCA_CASIA1of10 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config iresnet50_init-MS1M-ReCA_CASIA1of10 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config iresnet100_init-MS1M-ReCA_CASIA1of10 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config iresnet18_init-MS1M-ReCA_LampHQ1of10 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config iresnet50_init-MS1M-ReCA_LampHQ1of10 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config iresnet100_init-MS1M-ReCA_LampHQ1of10 | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config iresnet18_init-MS1M-ReCA_OuluCASIA | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config iresnet50_init-MS1M-ReCA_OuluCASIA | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config iresnet100_init-MS1M-ReCA_OuluCASIA | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config iresnet18_init-MS1M-ReCA_BUUA | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config iresnet50_init-MS1M-ReCA_BUUA | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config iresnet100_init-MS1M-ReCA_BUUA | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config mobilefacenet_init-MS1M-ReCA_BUUA | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config mobilefacenet_init-MS1M-ReCA_OuluCASIA | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.0" --master_port=1234  train_finetune_common_cls_RCT_relweights.py --config mobilefacenet_init-MS1M-ReCA_BUUA | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config mobilefacenet_init-MS1M-ReCA_LampHQ1of10 | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT_ortho.py --config mobilefacenet_init-MS1M-ReCA_LampHQ1of10 | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_ortho.py --config config_MobileNet_MS1M_ReCA | tee hist.log


python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config mobilefacenet_init-MS1M-ReCA_CASIA1of10 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config mobilefacenet_init-MS1M-ReCA_BUAA | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config mobilefacenet_init-MS1M-ReCA_OuluCASIA | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT.py --config mobilefacenet_init-MS1M-ReCA_LampHQ1of10 | tee hist.log


python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_subspace_ortho.py --config mobilefacenet_init-MS1M-ReCA_LampHQ1of10_subortho | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_2xmargin.py --config config_MobileNet_MS1M_ReCA | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT_meta.py --config mobilefacenet_init-MS1M-ReCA_LampHQ1of10 | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_ortho.py --config mobilefacenet_ms1m_0 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_ortho.py --config mobilefacenet_ms1m_1 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_ortho.py --config mobilefacenet_ms1m_2 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_ortho.py --config mobilefacenet_ms1m_3 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_ortho.py --config mobilefacenet_ms1m_4 | tee hist.log


python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_3repel.py --config mobilefacenet_ms1m_repel_0 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_3repel.py --config mobilefacenet_ms1m_repel_1 | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_ortho.py --config config_mobilenet_ms1m_ortho_4 | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_ortho.py --config config_mobilenet_ms1m_ortho_2 | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT2.py --config mobilefacenet_init-MS1M-ReCA_LampHQ1of10 | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_common_cls.py --config config_MobileNet_MS1M_ReCA_Combined | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_common_cls.py --config config_MobileNet_MS1M_ReCA_Combined | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  train_finetune_cls_RCT_dsbn.py --config mobilefacenet_dsbn_init-MS1M-ReCA_LampHQ1of10 | tee hist.log


python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_ortho.py --config config_LightCNN29_MS1M_ortho | tee hist.log


python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_ortho.py --config config_iresnet18_MS1M_ortho | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_ortho.py --config config_iresnet50_MS1M_ortho | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader_ortho.py --config config_iresnet100_MS1M_ortho | tee hist.log
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.3" --master_port=1236  finetune_MEC.py --config lightcnn29_init-MS1M_LAMP-HQ1of10_RCT | tee hist.log



python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_finetune_common_cls_RCT.py --config iresnet100_init-MS1M-ReCA_OuluCASIA.py | tee hist.log

python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader.py --config config_MobileNet_MS1M_ReCA | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_lightCNN9_MS1M | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.5" --master_port=1236  train_dataloader.py --config config_LightCNN29_MS1M | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_MobileNet_MS1M | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_iresnet50_MS1M | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_iresnet100_MS1M | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_iresnet100_MS1M_ReCA | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_MobileNet_MS1M | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_LightCNN29_MS1M  | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_iresnet18_MS1M  | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_iresnet18_MS1M_ReCA  | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_iresnet18_MS1M_RaCA  | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_iresnet100_MS1M_RaCA  | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_iresnet50_MS1M_RaCA  | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_MobileNet_MS1M_ReCA  | tee hist.log
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.2" --master_port=1235  train_dataloader.py --config config_MobileNet_MS1M_RaCA  | tee hist.log


