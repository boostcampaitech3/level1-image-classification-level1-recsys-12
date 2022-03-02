import json
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import wandb
from adamp import AdamP
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import Normalize, ToPILImage

from utils import *

from dataset import AgeBaseDataset, TestDataset, GenderBaseDataset, MaskOnlyBaseDataset
from loss import *


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_num_classes(task):
    if task == "gender":
        return 2
    elif task == "all":
        return 18
    else:
        return 3


def main(args):
    if args.wandb:
        log_list = ['epochs', 'lr', 'batch_size', 'criterion', 'optimizer', 'scheduler',
                    'augmentation']  # if add log hyperparameter args, add at this list
        config = {k: v for k, v in vars(args).items() if k in log_list}
        wandb.init(project='Ensembles', entity='kbp0237', name=args.name, config=config)

    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(args.model_dir, args.name), args)

    # --settings
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # -- dataset
    dataset = GenderBaseDataset(args.data_dir)
    num_classes = 2

    # -- augmentation
    transform_module = getattr(import_module('augmentation'), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
        p=args.flip_ratio
    )

    denormalize = transforms.Compose([
        Normalize(mean=[0., 0., 0.], std=[1 / 0.237, 1 / 0.247, 1 / 0.246]),
        Normalize(mean=[-0.548, -0.504, -0.479], std=[1., 1., 1.]),
        ToPILImage(),
    ])

    image_dir = os.path.join(args.test_data_dir, 'images')
    info_path = os.path.join(args.test_data_dir, 'info.csv')
    submission = pd.read_csv(info_path)
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    test_dataset = TestDataset(image_paths, resize=args.resize)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.valid_batch_size
    )

    fold_list = dataset.k_fold_split()
    oof_pred = None
    patience = 10
    counter = 0

    for fold in range(5):
        train_set, val_set = fold_list[fold]
        train_set.dataset.set_transform(transform)
        val_set.dataset.set_transform(transform)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count()//2,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True
        )

        model_module = getattr(import_module('model'), args.model)
        model = model_module(num_classes=num_classes, version=args.version).to(device)

        # -- criterion / optimizer
        criterion = create_criterion(args.criterion)
        # opt_module = getattr(import_module('torch.optim'), args.optimizer)
        optimizer = AdamP(
            params=model.parameters(),
            lr=args.lr,
            weight_decay=1e-5
        )
        sch_module = getattr(import_module('torch.optim.lr_scheduler'), args.scheduler)
        scheduler = sch_module(
            optimizer,
            **args.sch_params
        )

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=True, indent=4)

        wandb.watch(model)

        best_val_acc = 0
        best_val_loss = np.inf
        best_val_f1 = 0
        best_model = None
        print(f"Fold {fold + 1} Start!")
        for epoch in range(args.epochs):
            model.train()
            loss_value = 0
            matches = 0

            pred_list = []
            label_list = []

            train_f1 = 0
            avg_train_loss = 0
            for idx, batch in enumerate(train_loader):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=-1)

                pred_list.extend(preds.cpu().numpy())
                label_list.extend(labels.cpu().numpy())

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                avg_train_loss += loss.item()
                loss_value += loss.item()
                matches += (preds == labels).sum().item()

                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    train_f1 = f1_score(pred_list, label_list, average="macro")
                    current_lr = get_lr(optimizer)
                    print(
                        f"Fold {fold} - Epoch[{epoch + 1:3}/{args.epochs}]({idx + 1:>3}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4f} || "
                        f"training f1 score {train_f1:4.4f} || "
                        f"training accuracy {train_acc:4.2%} || "
                        f"lr {current_lr:4.6f}"
                    )

                    loss_value = 0
                    matches = 0

            scheduler.step()

            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                val_pred_list = []
                val_label_list = []

                for val_batch in val_loader:
                    inputs, val_label = val_batch
                    inputs, val_label = inputs.to(device), val_label.to(device)

                    val_outputs = model(inputs)
                    val_preds = torch.argmax(val_outputs, dim=-1)

                    loss_item = criterion(val_outputs, val_label)
                    acc_item = (val_label == val_preds).sum().item()

                    val_loss_items.append(loss_item)

                    val_acc_items.append(acc_item)

                    val_pred_list.extend(val_preds.cpu().numpy())
                    val_label_list.extend(val_label.cpu().numpy())

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                val_f1 = f1_score(val_pred_list, val_label_list, average='macro')
                best_val_loss = min(best_val_loss, val_loss)
                best_val_acc = max(best_val_acc, val_acc)

                if args.wandb:
                    total_train_acc = matches / len(train_set)
                    avg_train_loss = avg_train_loss / len(train_loader)
                    wandb.log({
                        f"Fold {fold + 1}": {"Avg Loss": {"train loss": avg_train_loss, "val loss": val_loss},
                                          "F1 Score": {"train f1 score": train_f1, "val f1 score": val_f1},
                                          "Accuracy": {"train acc": total_train_acc, "val acc": val_acc}},
                        "confusion mat": wandb.plot.confusion_matrix(preds=np.array(val_pred_list),
                                                                     y_true=np.array(val_label_list))
                        # "Wrong Images": diff_images
                    })

                if val_f1 > best_val_f1:
                    print(f"New best model for val f1 score : {val_f1:4.4f}! saving the best model..")
                    torch.save(model.state_dict(), f"{save_dir}/best.pth")
                    torch.save(model.state_dict(), f"{save_dir}/{epoch:03}_f1_{val_f1}.pth")
                    best_val_f1 = val_f1
                    best_model = model
                    counter = 0
                else:
                    counter += 1

                if counter > patience:
                    print("Early Stopping....")
                    break

                torch.save(model.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"[Val] f1: {val_f1:4.4f}, acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best f1: {best_val_f1:4.4f}, best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.4f}"
                )
                print()



        print("Evaluation Start!")
        all_predictions = []
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)

                pred = best_model(images)
                # pred += best_model(torch.flip(images, dims=(-1,))) / 2
                all_predictions.extend(pred.cpu().numpy())

            fold_pred = np.array(all_predictions)

        if oof_pred is None:
            oof_pred = fold_pred / 5
        else:
            oof_pred += fold_pred / 5
        print("Evaluation End")
        print()

    submission['ans'] = np.argmax(oof_pred, axis=1)
    submission.to_csv(os.path.join(args.output_dir, f'submission_gender_prediction_{args.model}_{args.version}.csv'), index=False)


if __name__ == '__main__':
    args = args_getter()

    main(args)
