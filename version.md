dataset: mnist
model: cleverhans pretrained
attack method: fgsm pgd deepfool

execute
```shell
git submodule update --init --recursive
```

libraries version
```tex
python 3.6.13
dm-tree==0.1.1
cleverhans 4.0.0
tensorflow 2.4.0
urllib3
```

执行参数
```shell
--model_name
carlini
--nb_examples
200
--attacks
"fgsm?eps=0.0156"
--robustness
"FeatureSqueezing?squeezer=bit_depth_1; FeatureSqueezing?squeezer=median_filter_2_2;"
--detection
"FeatureSqueezing?squeezers=bit_depth_1&distance_measure=l1&fpr=0.05; FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1&fpr=0.05; FeatureSqueezing?squeezers=median_filter_2_2&distance_measure=l1&fpr=0.05;"
```

