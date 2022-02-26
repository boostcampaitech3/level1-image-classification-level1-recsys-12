# Polar Model Experiment Logs

## Hyperparameter 설명

- `epochs` : 훈련에 진행한 epoch 수
- `batch_size` : batch size 크기
- `drop_out` : drop out layer가 존재한다면 ratio parameter
- `learning_rate` : learning rate

## Experiment Logs

| Date | Model name | Best loss | Avg Val Loss | Avg Val F1 Score | Best F1 Score | Hyperparameters |
|:--:|:---:|:---:|:---:|:---:|:---:|:---:|
|2022-02-23 14:20:17|ResNet18|0.0237|0.504|0.869|0.869|{epochs: 10, batch_size: 32, drop_out: 0.5, learning_rate: 0.01}|
|2022-02-23 14:55:51|ResNetNonPre|0.00918|0.974|0.771|0.85|{epochs: 10, batch_size: 32, drop_out: 0.5, learning_rate: 0.01}|
|2022-02-24 02:02:23|DenseNet|0.0385|0.434|0.434|0.774|{epochs: 10, batch_size: 32, drop_out: 0.5, learning_rate: 0.01}|
|2022-02-24 02:18:42|ResNet18_crop|0.00812|0.695|0.689|0.641|{epochs: 10, batch_size: 32, drop_out: 0.5, learning_rate: 0.01}|
|2022-02-24 10:28:34|ResNet18_crop|2.61e-08|0.857|0.816|0.76|{epochs: 100, batch_size: 64, drop_out: 0.5, learning_rate: 0.01}|
|2022-02-24 11:14:43|SENet_crop|0.000114|0.801|0.748|0.77|{epochs: 100, batch_size: 64, drop_out: 0.5, learning_rate: 0.001}|
|2022-02-24 11:38:23|SENet_NoCrop|2.91e-07|0.254|0.911|0.897|{epochs: 100, batch_size: 64, drop_out: 0.5, learning_rate: 0.001}|
|2022-02-24 12:14:38|SENet_NoCrop|1e+03|0.166|0.929|0.925|{epochs: 100, batch_size: 64, drop_out: 0.5, learning_rate: 0.001}|
|2022-02-24 22:22:12|MobileNetV2_4|1e+03|1.6|0.696|0.52|{epochs: 100, batch_size: 64, drop_out: 0.5, learning_rate: 0.001, wandb: true}|
|2022-02-24 23:18:25|SENet_4_cutmix|1e+03|0.688|0.721|0.736|{epochs: 100, batch_size: 32, drop_out: 0.5, learning_rate: 0.001, wandb: true, beta: 1.0}|
|2022-02-25 00:09:27|ResNet18_4_cutmix|1e+03|0.572|0.776|0.776|{epochs: 100, batch_size: 32, drop_out: 0.5, learning_rate: 0.001, wandb: true, beta: 1.0}|
