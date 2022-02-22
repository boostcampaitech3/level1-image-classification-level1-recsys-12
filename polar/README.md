# Polar Model Experiment Logs

## Hyperparameter 설명

- `epochs` : 훈련에 진행한 epoch 수
- `batch_size` : batch size 크기
- `d` : drop out layer가 존재한다면 ratio parameter
- `lr` : learning rate

| Date | Model name | Best loss | F1 Score | Avg Val Loss | Avf Val F1 Score | Hyperparameters |
|:--:|:---:|:---:|:---:|:---:|:---:|:---:|
|2022-02-22 08:07:21|ResNet18|0.00743|1|0.0176|0.0264|{epochs: 10, batch_size: 32, d: 0.5, lr: 0.01, name: ResNet18}|
