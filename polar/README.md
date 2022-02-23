# Polar Model Experiment Logs

## Hyperparameter 설명

- `epochs` : 훈련에 진행한 epoch 수
- `batch_size` : batch size 크기
- `drop_out` : drop out layer가 존재한다면 ratio parameter
- `learning_rate` : learning rate

## Experiment Logs

| Date | Model name | Best loss | Avg Val Loss | Avg Val F1 Score | Best F1 Score | Hyperparameters |
|:--:|:---:|:---:|:---:|:---:|:---:|:---:||2022-02-23 14:20:17|ResNet18|0.0237|0.504|0.504|                        0.869|0.869|{epochs: 10, batch_size: 32, drop_out: 0.5, learning_rate: 0.01}|
|2022-02-23 14:55:51|ResNetNonPre|0.00918|0.974|0.974|                        0.771|0.85|{epochs: 10, batch_size: 32, drop_out: 0.5, learning_rate: 0.01}|
|2022-02-24 02:02:23|DenseNet|0.0385|0.434|0.434|                        0.774|0.774|{epochs: 10, batch_size: 32, drop_out: 0.5, learning_rate: 0.01}|
|2022-02-24 02:18:42|ResNet18_crop|0.00812|0.695|0.695|                        0.689|0.641|{epochs: 10, batch_size: 32, drop_out: 0.5, learning_rate: 0.01}|
