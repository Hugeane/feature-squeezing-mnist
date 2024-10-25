version

```tex
branch py35-tf24

python 3.6.13
tensorflow 2.4.0
numpy 1.19.2
matplotlib 3.1.3
pillow 8.0.0
scikit-learn 0.22
urllib 1.26.8
cntk 2.5
```

script params
```shell
--dataset_name
CIFAR-10
--model_name
DenseNet
--nb_examples
10
--attacks
"fgsm?eps=0.0156;bim?eps=0.008&eps_iter=0.0012"
--robustness
"none; FeatureSqueezing?squeezer=bit_depth_5; FeatureSqueezing?squeezer=bit_depth_4; FeatureSqueezing?squeezer=median_filter_2_2; FeatureSqueezing?squeezer=non_local_means_color_13_3_4;"
--detection
" FeatureSqueezing?squeezers=bit_depth_1&distance_measure=l1&fpr=0.05; FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1&fpr=0.05; FeatureSqueezing?squeezers=bit_depth_3&distance_measure=l1&fpr=0.05; FeatureSqueezing?squeezers=bit_depth_4&distance_measure=l1&fpr=0.05; FeatureSqueezing?squeezers=bit_depth_5&distance_measure=l1&fpr=0.05; FeatureSqueezing?squeezers=median_filter_2_2&distance_measure=l1&fpr=0.05; FeatureSqueezing?squeezers=median_filter_3_3&distance_measure=l1&fpr=0.05; FeatureSqueezing?squeezers=non_local_means_color_11_3_2&distance_measure=l1&fpr=0.05; FeatureSqueezing?squeezers=non_local_means_color_11_3_4&distance_measure=l1&fpr=0.05; FeatureSqueezing?squeezers=non_local_means_color_13_3_2&distance_measure=l1&fpr=0.05; FeatureSqueezing?squeezers=non_local_means_color_13_3_4&distance_measure=l1&fpr=0.05; FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_13_3_2&distance_measure=l1&fpr=0.05;"
```
