import os
from datetime import datetime
import pytz

def record_expr(model_name, best_train_loss, avg_val_loss, avg_val_score, best_val_f1, args):
    # | Date | model_name | best_loss | avg val loss | avg val f1 score | best f1 | Hyperparameters |
    current_time = datetime.now(pytz.timezone('Asia/Seoul'))

    base_url = '/opt/ml'
    # model_state_save_path = base_url+f'/level1-image-classification-level1-recsys-12/polar/model_state/' \
    #                                  f'{model_name}_{current_time.month}{current_time.day}_' \
    #                                  f'{current_time.hour}{current_time.minute}.pt'
    markdown_path = base_url+'/Boostcamp-AI-Tech/README.md'
    # markdown_path = base_url+'/level1-image-classification-level1-recsys-12/seo_h2/README.md'
    # non_list = ['name', 'mode', 'save']
    # hypers = {k: v for k, v in vars(args).items() if k not in non_list}
    hypers = {k: v for k, v in vars(args).items()}
    current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    os.system(f"echo '|{current_time}|{model_name}|{best_train_loss:.3g}|{avg_val_loss:.3g}|"
              f"{avg_val_score:.3g}|{best_val_f1:.3g}|{str(hypers)}|' >> {markdown_path}")
    print("Save the experiment complete!")