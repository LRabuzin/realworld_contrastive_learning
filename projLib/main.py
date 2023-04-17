import argparse
import json
import os
import random
import uuid
import warnings
import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18

from losses import infonce_loss
from datasets import RealWorldIdentDataset
from infinite_iterator import InfiniteIterator
from pair_constructor import PairConfiguration

from tqdm import tqdm
import wandb

def collate_fn(batch):
        image1 = torch.stack([sample["image1"] for sample in batch])
        image2 = torch.stack([sample["image2"] for sample in batch])
        content = [sample["content"] for sample in batch]

        return {
            "image1": image1,
            "image2": image2,
            "content": content
            }

def train_step(data, encoder, loss_func, optimizer, params):
    if optimizer is not None:
        optimizer.zero_grad()
        encoder.train()
    else:
        encoder.eval()
        torch.set_grad_enabled(False)

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
    else:
        torch.set_grad_enabled(True)

    return loss_value.item()

def val_step(data, encoder, loss_func):
    return train_step(data, encoder, loss_func, optimizer=None, params=None)


def get_data(dataset, encoder, loss_func, dataloader_kwargs, content_categories):
    encoder.eval()
    loader = DataLoader(dataset, collate_fn=collate_fn, **dataloader_kwargs)
    rdict = {"hz_image_1": [], "hz_image_2": [],"loss_values": [], "labels": []}
    labels_dict = {category:[] for category in content_categories}

    with torch.no_grad():
        for data in loader:  # NOTE: can yield slightly too many samples
            loss_value = val_step(data, encoder, loss_func)
            rdict["loss_values"].append([loss_value])

            hz_image_1 = encoder(data["image1"])
            hz_image_2 = encoder(data["image2"])
            for i in range(len(hz_image_1)):
                rdict["hz_image_1"].append(hz_image_1[i].detach().cpu().numpy())
                rdict["hz_image_2"].append(hz_image_2[i].detach().cpu().numpy())
            for category in content_categories:
                zipped_content = zip(*[list(content) for content in data["content"]])
                labels_dict[category].extend([1 if category in zipped else 0 for zipped in zipped_content])
    rdict['labels'] = labels_dict
    return rdict


def evaluate_prediction(model, metric, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred), y_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--encoding-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--encoder-number", type=int, default=100000)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--leq-content-factors", action="store_true")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=500000)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--checkpoint-steps", type=int, default=100000)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--seed", type=int, default=np.random.randint(32**2-1))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--save-all-checkpoints", action="store_true")
    parser.add_argument("--load-from-memory", action="store_true")
    args = parser.parse_args()
    return args, parser

def main():
    args, _ = parse_args()

    if args.model_id is None:
        setattr(args, "model_id", str(uuid.uuid4()))
    args.save_dir = os.path.join(args.model_dir, args.model_id)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

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

    run = wandb.init(
        project="realworld-blockident",
        name=args.model_id,
        config=vars(args),
        tags=["baseline_encoding", "rn18"],)

    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()
    loss_func = lambda z1, z2: infonce_loss(
        z1, z2, sim_metric=sim_metric, criterion=criterion, tau=args.tau)
    
    mean_per_channel = [0.485, 0.456, 0.406]  # values from ImageNet
    std_per_channel = [0.229, 0.224, 0.225]   # values from ImageNet
    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean_per_channel, std_per_channel)])
    
    dataset_kwargs = {"transform": transform}
    dataloader_kwargs = {
        "batch_size": args.batch_size, "shuffle": True, "drop_last": True,
        "num_workers": args.workers, "pin_memory": True}
    
    # define dataloaders
    train_annotations = os.path.join(args.data_dir, "train.json")
    val_annotations = os.path.join(args.data_dir, "val.json")
    test_annotations = os.path.join(args.data_dir, "test.json")
    categories = os.path.join(args.data_dir, "categories.json")
    if args.leq_content_factors:
        ns = list(range(1, args.n+1))
    else:
        ns = [args.n]
    config = PairConfiguration([train_annotations, val_annotations, test_annotations], categories, k=args.k, n=ns)
    keep_in_memory = not args.load_from_memory
    dataset = RealWorldIdentDataset(args.data_dir, config.sample_pairs(), keep_in_memory=keep_in_memory, **dataset_kwargs)
    content_categories = config.content_categories

    train_len = math.floor(0.6*len(dataset))
    val_len = math.floor(0.2*len(dataset))
    test_len = math.floor(0.2*len(dataset))

    leftover_len = len(dataset) - train_len - val_len - test_len

    val_len += leftover_len

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_dataset, collate_fn = collate_fn, **dataloader_kwargs)
    train_iterator = InfiniteIterator(train_loader)
    
    # define encoder
    encoder = torch.nn.Sequential(
        resnet18(num_classes=args.hidden_size), # change to 34
        torch.nn.LeakyReLU(),
        torch.nn.Linear(args.hidden_size, args.encoding_size))
    encoder = torch.nn.DataParallel(encoder)
    encoder.to(device)

    wandb.watch(encoder, loss_func, 'all', 200)

    # for evaluation, always load saved encoders
    if args.evaluate:
        path_encoder = os.path.join(args.save_dir, f"encoder_{args.encoder_number}.pt")
        encoder.load_state_dict(torch.load(path_encoder, map_location=device))

    params = list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if not args.evaluate:
        step = 1
        loss_values = []
        with tqdm(total=args.train_steps) as pbar:
            while (step <= args.train_steps):

                data = next(train_iterator)
                loss_value = train_step(data, encoder, loss_func, optimizer, params)
                loss_values.append(loss_value)

                if step % args.log_steps == 1 or step == args.train_steps:
                    wandb.log({
                        "train/iteration": step,
                        "train/loss": loss_value,
                    })
                    print(f"Step: {step} \t",
                        f"Loss: {loss_value:.4f} \t",
                        f"<Loss>: {np.mean(loss_values[-args.log_steps:]):.4f} \t")

                if step % args.checkpoint_steps == 1 or step == args.train_steps:
                    torch.save(encoder.state_dict(), os.path.join(args.save_dir, f"encoder_{step}.pt"))
                    if args.save_all_checkpoints:
                        torch.save(encoder.state_dict(), os.path.join(args.save_dir, f"encoder_{step}.pt"))
                step += 1
                pbar.update(1)
            wandb.log_artifact(encoder)
    else:
        val_dict = get_data(val_dataset, encoder, loss_func, dataloader_kwargs, content_categories)
        test_dict = get_data(test_dataset, encoder, loss_func, dataloader_kwargs, content_categories)
        print(val_dict)
        print("*************")
        print(test_dict)

        print(f"<Val Loss>: {np.mean(val_dict['loss_values']):.4f}")
        print(f"<Test Loss>: {np.mean(test_dict['loss_values']):.4f}")

        results = []

        # select data
        train_inputs = np.concatenate((val_dict[f"hz_image_1"], val_dict[f"hz_image_2"]))
        test_inputs = np.concatenate((test_dict[f"hz_image_1"], test_dict[f"hz_image_2"]))
        train_labels = {category: np.concatenate((val_dict["labels"][category], val_dict["labels"][category])) for category in content_categories}
        test_labels = {category: np.concatenate((test_dict["labels"][category], test_dict["labels"][category])) for category in content_categories}
        data = [train_inputs, train_labels, test_inputs, test_labels]

        intermediates = []
        raw_predictions = {}
        for category in content_categories:
            mlpreg = MLPClassifier(max_iter=1000)
            acc_mlp, raw_prediction = evaluate_prediction(mlpreg, accuracy_score, data[0], data[1][category], data[2], data[3][category])
            intermediates.append(acc_mlp)
            raw_predictions[category] = [int(prediction) for prediction in raw_prediction]

        with open(os.path.join(args.save_dir, 'raw_preds.json'), 'w') as fp:
            json.dump(raw_predictions, fp)
        
        # append results
        results.append(intermediates)

        # convert evaluation results into tabular form
        columns = [f"acc_{int(category)}" for category in content_categories]
        df_results = pd.DataFrame(results, columns=columns)
        df_results.to_csv(os.path.join(args.save_dir, "results.csv"))
        print(df_results.to_string())


if __name__ == "__main__":
    main()