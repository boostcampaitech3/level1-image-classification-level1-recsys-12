import os
from datetime import datetime
import pytz


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


def record_expr(model, model_name, best_train_loss, best_train_score, avg_val_loss, avg_val_score, args):
    # | Date | model_name | best_loss | f1 score | avg val loss | avg val f1 score | Hyperparameters |
    base_url = '/opt/ml'
    model_state_save_path = base_url+'/level1-image-classification-level1-recsys-12/polar/model_state'
    markdown_path = base_url+'/level1-image-classification-level1-recsys-12/polar/README.md'
    current_time = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")

    os.system(f"echo '|{current_time}|{model_name}|{best_train_loss:.3g}|{best_train_score:.3g}|{avg_val_loss:.3g}|{avg_val_score:.3g}|{str(vars(args))}|' >> {markdown_path}")
    print("Save the experiment complete!")