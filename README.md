# Server-Side-Local-Gradient-Averaging-and-Learning-Rate-Acceleration-for-Scalable-Split-Learning

To run the code for experiments use the following in the command line:

```
!CUDA_VISIBLE_DEVICES=0 python train_data_mixup.py --dataset fmnist --model resnet --layers 16 --droprate 0.0 --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --skewness_ratio 0.5 --aux_net_config 1c2f --no-augment  --workers 9 --epoch 250 --batch_size 64 --local_label_size 500 --mixing_mode 'concat_batch' --lr 1e-3 --lr_pow 1.0 --comm_flag 0.6 --run 1 --extra 'final_5k__pow_1.0_adam_average_broadcast'
```

