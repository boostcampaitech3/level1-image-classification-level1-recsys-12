# Polar Model Experiment Logs

## Hyperparameter 설명

- `epochs` : 훈련에 진행한 epoch 수
- `batch_size` : batch size 크기
- `d` : drop out layer가 존재한다면 ratio parameter
- `lr` : learning rate

| Date | Model name | Best loss | F1 Score | Avg Val Loss | Avg Val F1 Score | Hyperparameters |
|:--:|:---:|:---:|:---:|:---:|:---:|:---:|
|2022-02-22 17:07:21|ResNet18|0.00743|1|0.0176|0.0264|{epochs: 10, batch_size: 32, d: 0.5, lr: 0.01, name: ResNet18}|
|2022-02-22 22:11:31|ResNet18|0.0285|1|0.454|0.77|{epochs: 10, batch_size: 32, d: 0.5, lr: 0.01, name: ResNet18, save: true}|
