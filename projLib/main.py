import argparse
import json
import os
import random
import uuid
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from losses import infonce_loss
from datasets import RealWorldIdent
from infinite_iterator import InfiniteIterator

def train_step(data, encoder, loss_func, optimizer, params):

    # reset grad
    if optimizer is not None:
        optimizer.zero_grad()

    # compute loss
    x1 = data['image1']
    x2 = data['image2']
    hz1 = encoder(x1)
    hz2 = encoder(x2)
    loss_value1 = loss_func(hz1, hz2)
    loss_value2 = loss_func(hz2, hz1)
    loss_value = 0.5 * (loss_value1 + loss_value2)  # symmetrized infonce loss

    # backprop
    if optimizer is not None:
        loss_value.backward()
        clip_grad_norm_(params, max_norm=2.0, norm_type=2)  # stabilizes training
        optimizer.step()

    return loss_value.item()

def val_step(data, encoder, loss_func):
    return train_step(data, encoder, loss_func, optimizer=None, params=None)


def get_data(dataset, encoder, loss_func, dataloader_kwargs):
    loader = DataLoader(dataset, **dataloader_kwargs)
    iterator = InfiniteIterator(loader)
    rdict = {"hz_image_1": [], "hz_image_2": [],"loss_values": [], "labels": []}
    i = 0
    with torch.no_grad():
        while (i < len(dataset)):  # NOTE: can yield slightly too many samples

            # load batch
            i += loader.batch_size
            data = next(iterator)  # contains images, texts, and labels

            # compute loss
            loss_value = val_step(data, encoder, loss_func)
            rdict["loss_values"].append([loss_value])

            # collect representations
            hz_image_1 = encoder(data["image1"])
            hz_image_2 = encoder(data["image2"])
            rdict["hz_image_1"].append(hz_image_1.detach().cpu().numpy())
            rdict["hz_image_2"].append(hz_image_2.detach().cpu().numpy())
            rdict["labels"] = data['content']

    # concatenate each list of values along the batch dimension
    # for k, v in rdict.items():
    #     rdict[k] = torch.tensor(v)

    return rdict


def evaluate_prediction(model, metric, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--image_pairs_train", type=str, required=True)
    parser.add_argument("--image_pairs_val", type=str, default='')# required=True)
    parser.add_argument("--image_pairs_test", type=str, default='')# required=True)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--encoding-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--log-steps", type=int, default=2)
    parser.add_argument("--checkpoint-steps", type=int, default=10000)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--seed", type=int, default=np.random.randint(32**2-1))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--save-all-checkpoints", action="store_true")
    args = parser.parse_args()
    return args, parser

def main():
    args, parser = parse_args()

    if args.model_id is None:
        setattr(args, "model_id", str(uuid.uuid4()))
    args.save_dir = os.path.join(args.model_dir, args.model_id)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    if not args.evaluate:
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as fp:
            json.dump(args.__dict__, fp)
    
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda"
    else:
        device = "cpu"
        warnings.warn("cuda is not available or --no-cuda was set.")

    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()
    loss_func = lambda z1, z2: infonce_loss(
        z1, z2, sim_metric=sim_metric, criterion=criterion, tau=args.tau)
    
    mean_per_channel = [0.4327, 0.2689, 0.2839]  # values from 3DIdent
    std_per_channel = [0.1201, 0.1457, 0.1082]   # values from 3DIdent
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_per_channel, std_per_channel)])
    
    dataset_kwargs = {"transform": transform}
    dataloader_kwargs = {
        "batch_size": args.batch_size, "shuffle": True, "drop_last": True,
        "num_workers": args.workers, "pin_memory": True}
    
    # define dataloaders
    train_dataset = RealWorldIdent(args.data_dir, args.image_pairs_train, **dataset_kwargs)
    if args.evaluate:
        val_dataset = RealWorldIdent(args.data_dir, args.image_pairs_val, **dataset_kwargs)
        test_dataset = RealWorldIdent(args.data_dir, args.image_pairs_test, **dataset_kwargs)
    else:
        train_loader = DataLoader(train_dataset, **dataloader_kwargs)
        train_iterator = InfiniteIterator(train_loader)

    
    # define encoder
    encoder = torch.nn.Sequential(
        resnet18(num_classes=args.hidden_size),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(args.hidden_size, args.encoding_size))
    encoder = torch.nn.DataParallel(encoder)
    encoder.to(device)

    # for evaluation, always load saved encoders
    if args.evaluate:
        path_encoder = os.path.join(args.save_dir, "encoder.pt")
        encoder.load_state_dict(torch.load(path_encoder, map_location=device))

    params = list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if not args.evaluate:

        # training loop
        step = 1
        loss_values = []  # list to keep track of loss values
        while (step <= args.train_steps):

            # training step
            data = next(train_iterator)  # contains images, texts, and labels
            loss_value = train_step(data, encoder, loss_func, optimizer, params)
            loss_values.append(loss_value)

            # print average loss value
            if step % args.log_steps == 1 or step == args.train_steps:
                print(f"Step: {step} \t",
                      f"Loss: {loss_value:.4f} \t",
                      f"<Loss>: {np.mean(loss_values[-args.log_steps:]):.4f} \t")

            # save models and intermediate checkpoints
            if step % args.checkpoint_steps == 1 or step == args.train_steps:
                torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder.pt"))
                if args.save_all_checkpoints:
                    torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder_%d.pt" % step))
            step += 1
    else:

        # collect encodings and labels from the validation and test data
        val_dict = get_data(val_dataset, encoder, loss_func, dataloader_kwargs)
        test_dict = get_data(test_dataset, encoder, loss_func, dataloader_kwargs)
        print(val_dict)
        print("*************")
        print(test_dict)

        # print average loss values
        print(f"<Val Loss>: {np.mean(val_dict['loss_values']):.4f}")
        print(f"<Test Loss>: {np.mean(test_dict['loss_values']):.4f}")

        # handle edge case when the encodings are 1-dimensional
        if args.encoding_size == 1:
            for m in ["image_1", "image_2"]:
                val_dict[f"hz_{m}"] = val_dict[f"hz_{m}"].reshape(-1, 1)
                test_dict[f"hz_{m}"] = test_dict[f"hz_{m}"].reshape(-1, 1)

        # # standardize the encodings
        # for m in ["image_1", "image_2"]:
        #     scaler = StandardScaler()
        #     val_dict[f"hz_{m}"] = np.squeeze(val_dict[f"hz_{m}"])
        #     test_dict[f"hz_{m}"] = np.squeeze(test_dict[f"hz_{m}"])
        #     val_dict[f"hz_{m}"] = scaler.fit_transform(val_dict[f"hz_{m}"])
        #     test_dict[f"hz_{m}"] = scaler.transform(test_dict[f"hz_{m}"])

        results = []

        # select data
        train_inputs = np.concatenate((val_dict[f"hz_image_1"], val_dict[f"hz_image_2"]))
        test_inputs = np.concatenate((test_dict[f"hz_image_1"], test_dict[f"hz_image_2"]))
        train_labels = np.concatenate((val_dict[f"labels"], val_dict[f"labels"]))
        test_labels = np.concatenate((test_dict[f"labels"], test_dict[f"labels"]))
        data = [train_inputs, train_labels, test_inputs, test_labels]

        # print(train_inputs)
        # print("*********************")
        # print(train_labels)
        # print("*********************")
        # print(test_inputs)
        # print("*********************")
        # print(test_labels)
        # print("*********************")

        acc_logreg, acc_mlp = [np.nan] * 2

        # logistic classification
        logreg = LogisticRegression(n_jobs=-1, max_iter=1000)
        acc_logreg = evaluate_prediction(logreg, accuracy_score, *data)
        # nonlinear classification
        mlpreg = MLPClassifier(max_iter=1000)
        acc_mlp = evaluate_prediction(mlpreg, accuracy_score, *data)

        # append results
        results.append([acc_logreg, acc_mlp])

        # convert evaluation results into tabular form
        columns = ["acc_logreg", "acc_mlp"]
        df_results = pd.DataFrame(results, columns=columns)
        df_results.to_csv(os.path.join(args.save_dir, "results.csv"))
        print(df_results.to_string())


if __name__ == "__main__":
    main()