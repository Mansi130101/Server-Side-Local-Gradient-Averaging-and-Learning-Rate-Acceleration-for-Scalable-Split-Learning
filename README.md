# Server-Side-Local-Gradient-Averaging-and-Learning-Rate-Acceleration-for-Scalable-Split-Learning

In the present repository, you can find the code for the following paper [[LINK]](https://github.com/Mansi130101/Server-Side-Local-Gradient-Averaging-and-Learning-Rate-Acceleration-for-Scalable-Split-Learning/blob/734eb2203ccc4d58e03a733a9b8044f72c2d8c8e/FL-AAAI-22_paper_21.pdf). It has been orally presented and published in AAAI Conference 2021. You can also find the paper globally available on [[LINK]](https://federated-learning.org/fl-aaai-2022/Papers/FL-AAAI-22_paper_21.pdf).

To give a brief overview of how to focus to tackle the problem of faster computation, reduced communication bandwidth, and higher privacy architectural changes to client-server model architecture with the proposed _split learning with a gradient averaging and learning rate splitting_ (**SGLR**) learning technique. It is a modification proposed of the present Split-Fed learning techniques, solving the 2 major issues of client-server imbalance while training, and privacy and communication bandwidth improvement. We thus present the respective solution of server-side **learning rate acceleration** and **gradient averaging** proving the solution along with scalable performance.

To run the code for experiments use the following in the command line:

```
!CUDA_VISIBLE_DEVICES=0 python train_data_mixup.py --dataset fmnist --model resnet --layers 16 --droprate 0.0 --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --skewness_ratio 0.5 --aux_net_config 1c2f --no-augment  --workers 9 --epoch 250 --batch_size 64 --local_label_size 500 --mixing_mode 'concat_batch' --lr 1e-3 --lr_pow 1.0 --comm_flag 0.6 --run 1 --extra 'final_5k__pow_1.0_adam_average_broadcast'
```

Below is the figure from the paper, depicting the various previously used learning architectures and the solutions proposed in the paper:
![image](https://github.com/Mansi130101/Server-Side-Local-Gradient-Averaging-and-Learning-Rate-Acceleration-for-Scalable-Split-Learning/assets/58467251/f0b426df-18fc-4ee9-b919-c78c10186568)

For future works, studies on Extending SplitLr: cyclic learning rate splitting method, Convergence Analysis, and Differential Privacy Analysis on SplitAvg can be taken ahead.

