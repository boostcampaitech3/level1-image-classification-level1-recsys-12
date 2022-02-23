import os
from datetime import datetime
import pytz
import pandas as pd
import argparse
import re


def args_getter():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=10, type=int, help='Num train epochs (default=10)')
    parser.add_argument("-bs", "--batch_size", default=32, type=int, help="Num of batch size (default=32)")
    parser.add_argument("-d", "--drop_out", default=0.5, type=float, help="dropout ratio (default=0.5)")
    parser.add_argument("-lr", "--learning_rate", default=0.01, type=float, help="Learning rate (default=0.01)")
    parser.add_argument("-m", "--mode", default="train", type=str, help="Training mode or Test mode")
    parser.add_argument("-n", "--name", default="ResNet18", help="Model name (if you use pre-trained, \
                                                                            write pre-trained model name)")
    parser.add_argument("-s", "--save", default="false", help="Save the experiments")
    # parser.add_argument("--m", required=False, default=None, type=float, help="momentum (default=0.9)")

    args = parser.parse_args()

    return args


def make_img_path_df(path, output_dir=None, train=True):
    data_path = '/opt/ml/input/data'

    if train:
        df = pd.DataFrame(None, columns=['path', 'class'])
        data_path += '/train/train.csv'
        train_df = pd.read_csv(data_path)

        for idx, row in enumerate(train_df.iloc):
            for file in list(os.listdir(os.path.join(path, row['path']))):
                if file[0] == '.':
                    continue
                else:
                    file_name = file.split('.')[0]

                    if file_name == "normal":
                        mask = 2
                    elif file_name == "incorrect_mask":
                        mask = 1
                    else:
                        mask = 0
                gender = 0 if row['gender'] == "male" else 1
                data = {
                    'path': os.path.join(path, row['path'], file),
                    'class':mask*6 + gender*3 + min(2, row['age']//30)
                }
                df = df.append(data, ignore_index=True)

        df.to_csv(output_dir, index=False)


def make_info_df(img_path_df, output_dir=None):
    new_df = pd.DataFrame(None, columns=['id', 'gender', 'age', 'types', 'path', 'class'])

    for _, data in enumerate(img_path_df.iloc):
        path = data['path']
        path_split = path.split('/')
        user_id = path_split[7].split('_')[0]
        gender = path_split[7].split('_')[1]
        age = int(path_split[7].split('_')[3])
        mask = re.findall(r'[a-z]*', path_split[8].split('.')[0])[0]
        mask_type = None
        if mask == "normal":
            mask_type = 2
        elif mask == "incorrect":
            mask_type = 1
        else:
            mask_type = 0

        new_data = {
            'id': user_id,
            'gender': 0 if gender == "male" else 1,
            'age': age//30,
            'types': mask_type,
            'path': path,
            'class': data['class']
        }

        new_df = new_df.append(new_data, ignore_index=True)

    new_df.to_csv(output_dir, index=False)


def record_expr(model, model_name, best_train_loss, avg_val_loss, avg_val_score, best_val_f1, args):
    # | Date | model_name | best_loss | avg val loss | avg val f1 score | best f1 | Hyperparameters |
    current_time = datetime.now(pytz.timezone('Asia/Seoul'))

    base_url = '/opt/ml'
    model_state_save_path = base_url+f'/level1-image-classification-level1-recsys-12/polar/model_state/' \
                                     f'{model_name}_{current_time.month}{current_time.day}_' \
                                     f'{current_time.hour}{current_time.minute}.pt'
    markdown_path = base_url+'/level1-image-classification-level1-recsys-12/polar/README.md'
    non_list = ['name', 'mode', 'save']
    hypers = {k: v for k, v in vars(args).items() if k not in non_list}

    current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    os.system(f"echo '|{current_time}|{model_name}|{best_train_loss:.3g}|{avg_val_loss:.3g}|{avg_val_loss:.3g}|\
                        {avg_val_score:.3g}|{best_val_f1:.3g}|{str(hypers)}|' >> {markdown_path}")
    print("Save the experiment complete!")