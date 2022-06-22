from ast import arg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
class Mydata(Dataset):
    def __init__(self, csv_file_path, args):
        self.csv_file_path = csv_file_path
        self.args = args
    def __getitem__(self, index):
        label, feats=get_bag_feats(self.csv_file_path.iloc[index], self.args)
        label = torch.Tensor(label).float()
        feats = torch.Tensor(feats).float()
        # print(label)
        return label, feats
    def __len__(self):
        return len(self.csv_file_path)

def get_bag_feats(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = csv_file_df.iloc[0]
        # feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    feats = feats[:, :args.feats_size]
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
        
    return label, feats

def train(train_df, milnet, criterion, optimizer, args):
    milnet.train()
    csvs = shuffle(train_df).reset_index(drop=True)
    total_loss = 0
    bc = 0
    Tensor = torch.cuda.FloatTensor
    data_train = Mydata(train_df, args)
    train_loader = DataLoader(data_train, batch_size = 1, num_workers = 10,shuffle = False)
    # for i in range(len(train_df)):
    for i, datai in enumerate(train_loader):
        optimizer.zero_grad()
        # label, feats = get_bag_feats(train_df.iloc[i], args)
        label, feats = datai
        # print(label)
        # print(feats)
        feats = dropout_patches(feats, args.dropout_patch)
        feats = Tensor(feats).unsqueeze(0)
        # bag_label = Variable(Tensor([label]))
        # bag_feats = Variable(Tensor([feats]))
        # bag_label = Variable(label)
        # bag_feats = Variable(feats)
        bag_label = label.cuda()
        bag_feats = feats.cuda()
        bag_feats = bag_feats.view(-1, args.feats_size)
        # ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        bag_prediction = milnet(bag_feats)
        # max_prediction, _ = torch.max(ins_prediction, 0)
        # print(bag_prediction) 
        # print(max_prediction)       
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        # max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        # loss = 0.5*bag_loss + 0.5*max_loss
        loss = bag_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)

def dropout_patches(feats, p):
    feats = feats.squeeze(0)
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats

def test(test_df, milnet, criterion, optimizer, args):
    milnet.eval()
    csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    data_test = Mydata(test_df, args)
    test_loader = DataLoader(data_test, batch_size = 1, num_workers = 10,shuffle = False)
    with torch.no_grad():
        for i, datai in enumerate(test_loader):
            label, feats = datai
            # bag_label = Variable(Tensor([label]))
            # bag_feats = Variable(Tensor([feats]))
            bag_label = label.cuda()
            bag_feats = feats.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)
            # ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            bag_prediction = milnet(bag_feats)
            # max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            # max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            # loss = 0.5*bag_loss + 0.5*max_loss
            loss = bag_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([label.squeeze(0).numpy()])
            # test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil
    
    # i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    # b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node).cuda()
    milnet = mil.MAXNet(args.feats_size, args.num_classes).cuda()
    # milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    # if args.model == 'dsmil':
    #     state_dict_weights = torch.load('init.pth')
    #     milnet.load_state_dict(state_dict_weights, strict=False)
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    
    if args.dataset == 'TCGA-lung-default':
        bags_csv = '20-TCGA-lung.csv'
        # bags_csv = 'datasets/TCGA-lung/TCGA-lung.csv'
    else:
        bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')
        
    bags_path = pd.read_csv(bags_csv)
    train_path = bags_path.iloc[0:int(len(bags_path)*(1-args.split)), :]
    test_path = bags_path.iloc[int(len(bags_path)*(1-args.split)):, :]
    best_score = 0
    best_auc_1 = 0
    best_auc_2 = 0
    best_acc = 0
    save_path = os.path.join('weights', datetime.date.today().strftime("%m%d%Y"))
    os.makedirs(save_path, exist_ok=True)
    run = len(glob.glob(os.path.join(save_path, '*.pth')))
    for epoch in range(1, args.num_epochs):
        train_path = shuffle(train_path).reset_index(drop=True)
        test_path = shuffle(test_path).reset_index(drop=True)
        train_loss_bag = train(train_path, milnet, criterion, optimizer, args) # iterate all bags
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_path, milnet, criterion, optimizer, args)
        # if args.dataset=='TCGA-lung-default':
        if args.dataset=='TCGA-lung-defaul':
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
        else:
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        scheduler.step()
        if avg_score > best_acc:
            best_acc = avg_score
            print('Best acc:%.4f'%(best_acc))
        if aucs[0] > best_auc_1:
            best_auc_1 = aucs[0]
            print('Best_auc_1:%.4f'%(best_auc_1))
        if aucs[1] > best_auc_2:
            best_auc_2 = aucs[1]
            print('Best_auc_2:%.4f'%(best_auc_2))
        current_score = (sum(aucs) + avg_score + 1 - test_loss_bag)/4
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(save_path, str(run+1)+'.pth')
            torch.save(milnet.state_dict(), save_name)
            # if args.dataset=='TCGA-lung-default':
            if args.dataset=='TCGA-lung-defaul':
                print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
            else:
                print('Best model saved at: ' + save_name)
                print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
            

if __name__ == '__main__':
    main()