# Code to run the ViT Transformer on CIFAR10
# all together in one document to make it more convenient for Peregrine.

import numpy as np
import json
import os.path
import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torchvision import transforms

from sklearn.model_selection import KFold

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention2(d2l.MultiHeadAttention):
    """Modification in the Multi-head attention to allow to project the attention
    into a larger space."""
    def __init__(self, dim_attention, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(num_hiddens, num_heads, dropout, bias, **kwargs)
        self.W_q = nn.LazyLinear(dim_attention, bias=bias)
        self.W_k = nn.LazyLinear(dim_attention, bias=bias)
        self.W_v = nn.LazyLinear(dim_attention, bias=bias)

class PatchEmbedding(nn.Module):
    """Class in charge of splitting the image into patches and embedding."""
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
    
class ViTMLP(nn.Module):
    """Class in charge of applying the FFN layer at the end of each block."""
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))
    
class ViTBlock(nn.Module):
    """Encoder block: applies attention and MLP."""
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        #num_hiddens = norm_shape
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = self.ln1(X)
        return X + self.mlp(self.ln2(
            X + self.attention(X, X, X, valid_lens)))
    
class ViT(d2l.Classifier):
    """Vision transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10, usewandb=False, optimizer_name="SGD"):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        #(batch size, no. of patches, num_hiddens)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        #(batch size, no. of patches + 1, num_hiddens)
        X = self.dropout(X + self.pos_embedding)
        #same structure
        for blk in self.blks:
            X = blk(X)
        #same structure
        return self.head(X[:, 0])
    
    def vis_attention(self, X):
        # Function to visualize the attention weights
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return X
    
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        #self.plot('loss', l, train=True)
        a = self.accuracy(self(*batch[:-1]), batch[-1])
        #wandb.log({"train_loss": l, "train_acc": a})
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        a = self.accuracy(self(*batch[:-1]), batch[-1])
        #wandb.log({"val_loss": l, "val_acc": a})
        #self.plot('loss', l, train=False)
        
    def configure_optimizers(self):
        """Extremely dangerous function if used maliciously.
        Its purpose is to allow to change the optimizer easily."""
        
        return eval("torch.optim."+ self.optimizer_name + "(self.parameters(), lr=self.lr)")
    
def evaluate_loss_gpu(net, data_iter, device=None):
    """Compute the loss for a model on a dataset using a GPU.
    I think this function is not used in the final product."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(net.loss(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
        
class Trainer2(d2l.Trainer):
    """Trainer to train the model. It is adapted from the Dive 2 Deep Learning one
    but I have made some modifications."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0, early_stop=3):
        super().__init__(max_epochs, num_gpus, gradient_clip_val)
        self.save_hyperparameters()
        
    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        best_loss = float('inf')
        counter = 0
        for self.epoch in range(self.max_epochs):
            loss = self.fit_epoch()
            if loss > best_loss:
                counter += 1
                #if counter >= self.early_stop:
                    #return self.epoch
            else:
                best_loss = loss
                counter = 0
        return self.epoch
        
    def fit_epoch(self):
        super().fit_epoch()
        acc = d2l.evaluate_accuracy_gpu(self.model, self.val_dataloader)
        loss = evaluate_loss_gpu(self.model, self.val_dataloader)
        wandb.log({"accuracy": acc, "loss": loss})
        print("Epoch: " + str(self.epoch) + " - acc: " + str(acc) + " - loss: " + str(loss))
        return loss


class CIFAR10(d2l.DataModule):
    """Envelope class to obtain the Dataloaders for CIFAR10 with some pretreatment"""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()

        # Image augmentation
        transform_train = transforms.Compose([
            transforms.Resize(40),
            transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                         ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2023, 0.1994, 0.2010])])
        
        transform_test = torchvision.transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2023, 0.1994, 0.2010])])
        
        #To choose depending on where you execute this.
        #self.train = torchvision.datasets.CIFAR10(root="/Users/joeldvd/Documents/uni/tfg", train=True, transform=transform_train, download=False)
        #self.val = torchvision.datasets.CIFAR10(root="/Users/joeldvd/Documents/uni/tfg", train=False, transform=transform_test, download=False)
        self.train = torchvision.datasets.CIFAR10(root="/data/s5091217", train=True, transform=transform_train, download=False)
        self.val = torchvision.datasets.CIFAR10(root="/data/s5091217", train=False, transform=transform_test, download=False)
        
        
    def text_labels(self, indices):
        """Return text labels."""
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = *batch[:-1], batch[-1]
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(1), nrows, ncols, titles=labels)
        

        
        
class CIFAR10_KFOLD(CIFAR10):
    """Modificaction of the previous class to allow for kfold over the train data"""
    def __init__(self, batch_size=64, resize=(28, 28), k=5, shuffle=True):
        super().__init__(batch_size, resize)
        self.save_hyperparameters()
        
        # Normally  k = 5
        self.k = k
        if k != 1:
            kf = KFold(n_splits=k)
            self.data = self.train # we use the train data
            self.g = kf.split(self.data)
            self.train_sampler, self.valid_sampler = next(self.g)
            np.random.shuffle(self.train_sampler)
        
        
    def get_dataloader(self, train):
        if self.k != 1:
            if train:
                train_loader = torch.utils.data.DataLoader(
                    self.data, self.batch_size,
                    num_workers=self.num_workers, sampler=self.train_sampler,
                )
                return train_loader
            else:
                valid_loader = torch.utils.data.DataLoader(
                    self.data, self.batch_size,
                    num_workers=self.num_workers, sampler=self.valid_sampler,
                )
                return valid_loader
        else:
            data = self.train if train else self.val
            return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
    
    def next(self):
        # Advance to the next k split
        if self.k != 1:
            self.train_sampler, self.valid_sampler = next(self.g)
                
class Measures():
    """
    Class for handling of statistics with ground truths
    """

    def __init__(self, model, data):
        # For CIFAR10
        model.eval()
        self.object_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        self.totals = [0] * len(self.object_names)  # the number of occurrences for each object type

        # Split in 10 categories, one for each class:
        self.tp = torch.zeros(len(self.object_names))  # true positives
        self.fp = torch.zeros(len(self.object_names))  # false positives
        self.tn = torch.zeros(len(self.object_names))  # true negatives
        self.fn = torch.zeros(len(self.object_names))  # false negatives
        
        test_data = data.get_dataloader(False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for batch in test_data:
            Y_hat = torch.Tensor(batch[0]).to(device)
            out = model(Y_hat)
            predictions = torch.argmax(out, dim=1)
            self.tn += len(batch[-1])

            ground_truths = batch[-1]
                    
            for pred, real in zip(predictions, ground_truths):
                self.totals[real] += 1
                
                # manually counting each instance
                if pred == real:
                    self.tp[real] += 1
                    self.tn[real] -= 1
                if pred != real:
                    self.fp[pred] += 1
                    self.fn[real] += 1
                    self.tn[real] -= 1
                    self.tn[pred] -= 1
                    

    def precision(self, i):
        if self.tp[i] + self.fp[i] == 0:
            return 0
        return self.tp[i] / (self.tp[i] + self.fp[i])

    def recall(self, i):
        if self.tp[i] + self.fn[i] == 0:
            return 0
        return self.tp[i] / (self.tp[i] + self.fn[i])

    def f_score(self, i):
        if self.precision(i) + self.recall(i) == 0:
            return 0
        return 2 * self.precision(i) * self.recall(i) / (self.precision(i) + self.recall(i))

    def macro_avg(self, function):
        return sum(function(i) for i in range(len(self.object_names))) / len(self.object_names)

    def weighted_avg(self, function):
        if sum(self.totals) == 0:
            return 0
        return sum(function(i) * self.totals[i] for i in range(len(self.object_names))) / sum(self.totals)

    def accuracy(self):
        if sum(self.totals) == 0:
            return 0
        return sum(self.tp) / sum(self.totals)
    
    def to_json(self):
        """
        returns json containing a sum up of the statistics measured
        """
        output = {}
        output['precision'] = self.macro_avg(self.precision).item()
        output['recall'] = self.macro_avg(self.recall).item()
        output['f_score'] = self.macro_avg(self.f_score).item()
        output['accuracy'] = self.accuracy().item()
        return output
    
    def to_json_pro(self):
        """
        returns json containing all the statistics measured
        """
        output = {}
        for obj, count in zip(self.object_names, self.totals):
            output[obj] = count
        output['precision'] = {}
        for i in range(len(self.object_names)):
            output['precision'][self.object_names[i]] = self.precision(i)
        output['precision']['macro_avg'] = self.macro_avg(self.precision)
        output['precision']['weighted_avg'] = self.weighted_avg(self.precision)
        output['recall'] = {}
        for i in range(len(self.object_names)):
            output['recall'][self.object_names[i]] = self.recall(i)
        output['recall']['macro_avg'] = self.macro_avg(self.recall)
        output['recall']['weighted_avg'] = self.weighted_avg(self.recall)
        output['f_score'] = {}
        for i in range(len(self.object_names)):
            output['f_score'][self.object_names[i]] = self.f_score(i)
        output['f_score']['macro_avg'] = self.macro_avg(self.f_score)
        output['f_score']['weighted_avg'] = self.weighted_avg(self.f_score)
        output['accuracy'] = self.accuracy()
        return output
                
                
def training(name="Vit_scratch", k=1, batch_size=128, optimizer_name="SGD", lr=0.1, maxEpochs=20,
             img_size=32, patch_size=4, num_hiddens=512, mlp_num_hiddens=2048,
             num_heads=8, num_blks=2, emb_dropout=0.1, blk_dropout=0.1, usewandb=True, project_name="Testing"):
    """Function that creates a module, trains it and saves the results in the history file"""

    trainer = Trainer2(max_epochs=maxEpochs, num_gpus=1)

    data = CIFAR10_KFOLD(batch_size=batch_size, resize=(img_size, img_size), k=k)
    
    if usewandb:
        wandb.login()
    
    scores = []
    for i in range(k):
        if usewandb:
            wandb.init(
                # Set the project where this run will be logged
                project=project_name, 
                # We pass a run name
                name=name, 
                # Track hyperparameters and run metadata
                config={
                    "architecture": "VITscratch",
                    "dataset": "CIFAR-10",
                    "kfold":k,
                    "batch_size": batch_size,
                    "optimizer_name": optimizer_name,
                    "learning_rate": lr,
                    "maxEpoch": maxEpochs,
                    "img_size": img_size,
                    "patch_size": patch_size,
                    "num_hiddens": num_hiddens,
                    "mlp_num_hiddens": mlp_num_hiddens, 
                    "num_heads":num_heads,
                    "num_blks":num_blks,
                    "emb_dropout": emb_dropout, 
                    "blk_dropout": blk_dropout
                })

    
    
        model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
                num_blks, emb_dropout, blk_dropout, lr, usewandb=usewandb, optimizer_name=optimizer_name)
        epochs = trainer.fit(model, data)
        scores.append(d2l.evaluate_accuracy_gpu(model, data.get_dataloader(False)))
        data.next()
        
        print("Acabado de entrenar "+ name + " en la epoch "+ str(epochs))
        if usewandb:
            wandb.finish()
        
    accuracy = np.mean(scores)
    print("Scores: ", scores)
    print("Mean performance: ", accuracy)

    


    # Save the results in a JSON
    history_path = '/data/s5091217/code_ego4d/history2.json'
    
    m = Measures(model, data)
    
    print(m.to_json())

    results = {
        "name":name,
        "architecture": "VITscratch",
        "dataset": "CIFAR-10",
        "kfold":k,
        "batch_size": batch_size,
        "optimizer_name": optimizer_name,
        "learning_rate": lr,
        "maxEpoch": maxEpochs,
        "epochs": epochs, # it will be from the last iteration of the kfold
        "img_size": img_size,
        "patch_size": patch_size,
        "num_hiddens": num_hiddens,
        "mlp_num_hiddens": mlp_num_hiddens, 
        "num_heads":num_heads,
        "num_blks":num_blks,
        "emb_dropout": emb_dropout, 
        "blk_dropout": blk_dropout,
        "accuracy": accuracy,
        "measures": str(m.to_json())
        }

    if not os.path.exists(history_path):

        data = {0: results}

        with open(history_path, "w") as f:
            json.dump(data, f, indent = 6)

    else:
        with open(history_path, "r") as f:
            data = json.load(f)

        data[len(data)] = results

        with open(history_path, "w") as f:
            json.dump(data, f)
            
    print(name + " trained correctly with score " + str(accuracy))
    
    return (model, accuracy)


#EXECUTION
for i in {4, 5, 6}:
    name1= "ViT-Base_blocks=" + str(i)
    name2= "ViT-Large_blocks=" + str(i)
    name3= "ViT-Huge_blocks=" + str(i)
    training(name=name1, maxEpochs=100, num_hiddens=768, mlp_num_hiddens=3072,
         num_heads=12, num_blks=i, optimizer_name="Adam", lr=0.0001, project_name="Testing_ViTs_final2")
    training(name=name2, maxEpochs=100, num_hiddens=1024, mlp_num_hiddens=4096,
         num_heads=16, num_blks=i, optimizer_name="Adam", lr=0.0001, project_name="Testing_ViTs_final2")
    training(name=name3, maxEpochs=100, num_hiddens=1280, mlp_num_hiddens=5120,
         num_heads=16, num_blks=i, optimizer_name="Adam", lr=0.0001, project_name="Testing_ViTs_final2")
