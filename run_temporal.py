import os, torch, random, json, gc
import numpy as np
import matplotlib.pyplot as plt 
from torchvision import transforms, models
from torch.utils.data import DataLoader
import argparse
from OASIS_2D.dataset import OASIS_TemporalDataset, collate_fn
from OASIS_2D.oasis_temporal import Experiment_Temporal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

vit_transforms = transforms.Compose([
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
    # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    transforms.ToTensor(), 
    # transforms.RandomResizedCrop(224),
    transforms.Resize(224),
    # transforms.RandomHorizontalFlip(),
])

model_dict = {
    'ResNet': models.resnet18,
    'ViT': models.vision_transformer.vit_b_16
}

class DNN(torch.nn.Module):
    def __init__(
        self, input_size=512, seq_len=3, num_layers=1,
        hidden_size=64, dropout=0.1
    ):
        super(DNN, self).__init__()
        # input shape is (batch, seq_len, features)
        self.flatten = torch.nn.Flatten()
        self.input = torch.nn.Linear(input_size*seq_len, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size)
        
        # list doesn't work here
        layers = torch.nn.ModuleList()
        for i in range(num_layers-1):
            layer = torch.nn.Linear(hidden_size, hidden_size)
            layers.append(layer)
        self.layers = layers
        
        self.projection = torch.nn.Linear(hidden_size, 2)
        # self.fc2 = torch.nn.Linear(16, 1)
        
    def forward(self, x, mask):
        # The output of nn.LSTM() is a tuple. The first 
        # element is the generated hidden states, 
        # one for each time step of the input. The 
        # second element is the LSTM cell’s memory 
        # and hidden states, which is not used here.
        # output (batch x hidden_size), (hc, cn)
        x = self.input(self.flatten(x))
        
        for i in range(len(self.layers)):
            x = self.layers[i](self.dropout(self.bn(x)))
        
        x = self.projection(self.dropout(self.bn(x)))
        # x = self.fc2(self.dropout(x))
        
        return x
    
class LstmModel(torch.nn.Module):
    def __init__(
        self, input_size=512, num_layers=1,
        hidden_size=64, dropout=0.1
    ):
        super(LstmModel, self).__init__()
        # input shape is (batch, seq_len, features)
        self.lstm = torch.nn.LSTM(
            input_size=input_size, 
            hidden_size= hidden_size, 
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, 1)
        # self.fc2 = torch.nn.Linear(16, 1)
        
    def forward(self, x, mask):
        # The output of nn.LSTM() is a tuple. The first 
        # element is the generated hidden states, 
        # one for each time step of the input. The 
        # second element is the LSTM cell’s memory 
        # and hidden states, which is not used here.
        # output (batch x hidden_size), (hc, cn)
        x, _ = self.lstm(x) 
        
        # The output of hidden states is further processed by a 
        # fully-connected layer to produce a single regression result. 
        # Since the output from LSTM is one per each input time step, 
        # you can chooce to pick only the last timestep’s output
        # x = self.fc(x[:, -1, :])
        
        # zero out the out for padded time steps
        # x = x * mask.unsqueeze(-1)
        
        end_index = mask.sum(dim=1).type(torch.long)-1
        x = x[range(len(x)), end_index]
        
        x = self.fc1(self.dropout(self.bn(x)))
        # x = self.fc2(self.dropout(x))
        
        return x

def main(args, itr_seed=None):
    original_seed = args.seed
    # set random seed
    if itr_seed is None:
        SEED = args.seed
        print(f'Random seed {SEED}')
        seed_everything(SEED)
        experiment_seeds = np.random.randint(1e3, size=args.itrs, dtype=int)
        print(f'Experiment seeds {experiment_seeds}.')
        
        # running an iteration directly from run
        if args.itr_num is not None:
            SEED = args.seed = int(experiment_seeds[args.itr_num-1])
            print(f'{args.itr_num}-th iteration using seed {SEED}.')
            
    else:
        # running an iteration from the whole experiment
        SEED = args.seed = itr_seed
        print(f'{args.itr_num}-th iteration using seed {SEED}.')
        seed_everything(SEED)
    
    # set result directory and call iterations
    if args.itr_num is not None:
        assert args.itrs is not None and args.itr_num <= args.itrs, 'itr_num must be smaller than itrs'
        result_dir = os.path.join('OASIS_2D', args.result_dir, f'{args.model}_seq_{args.seq_len}', str(args.itr_num))
        
    elif args.itrs is not None:
        for itr_num in range(1, args.itrs+1):
            args.itr_num = itr_num
            # args.seed = int(experiment_seeds[itr_num-1])
            main(args, int(experiment_seeds[itr_num-1]))
        return
    else:
        result_dir = os.path.join('OASIS_2D', args.result_dir, f'{args.model}_seq_{args.seq_len}')
    
    # Get cpu, gpu device for training.
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    feature_dir = os.path.join('OASIS_2D', args.result_dir, args.model)
    features = torch.load(os.path.join(feature_dir, 'features.pt'))
    dimension = len(features['feature'][0])
    print(f'Extracted feature has dimension {dimension}')
    
    # Prepare dataset and dataloader
    train_dataset = OASIS_TemporalDataset(
        features, train=True, seed=SEED
    )
    test_dataset = OASIS_TemporalDataset(
        features, train=False, seed=SEED
    )
    

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False, # shuffling not necessary for test
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    )
    
    # result_dir = os.path.join('OASIS_2D', 'results', 'ResNet18')
    exp = Experiment_Temporal(result_dir=result_dir, device=device)
    model = DNN(
        input_size=dimension, 
        seq_len=args.seq_len, hidden_size=args.hidden_size, 
        dropout=args.dropout, num_layers=1
    )
    model = model.float().to(device)
    
    if not args.test:
        train_history = exp.train(
            model, train_dataloader=train_dataloader, 
            val_dataloader=test_dataloader,
            epochs=args.epochs, learning_rate=args.lr
        )
        # Plot train history
        exp.plot_history(train_history)
    
    # Evaluate metrics
    print('Evaluating train data')
    train_result = exp.test(model, train_dataloader)

    print('Evaluating test data')
    test_result = exp.test(model, test_dataloader)
    
    # Dump evaluation results
    with open(os.path.join(result_dir, 'train.json'), 'w') as output_file:
        json.dump(train_result, output_file, indent=4)
        
    with open(os.path.join(result_dir, 'test.json'), 'w') as output_file:
        json.dump(test_result, output_file, indent=4)
    with open(os.path.join(result_dir, 'config.json'), 'w') as output_file:
        json.dump(vars(args), output_file, indent=4)
    
    args.seed = original_seed
    print('Done\n\n')

def get_parser():
    parser = argparse.ArgumentParser(
        description='Train model', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--itrs', default=5, type=int, help='number of iteration')
    parser.add_argument('--itr_num', default=None, type=int, help='iteration number. 1 <= itr_num <= itrs')
    # basic config
    parser.add_argument('--model', default='ResNet', type=str, help='model name', choices=list(model_dict.keys()))
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    parser.add_argument('--result_dir', default='results',  type=str, help='directory to save the result')
    parser.add_argument('--test', action='store_true', help='load a pretrained model for testing a split')
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    
    # model arguments
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--seq_len', default=3, type=int, help='sequence length')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.01, type=float, help='dropout rate')
    parser.add_argument('--hidden_size', default=64, type=int, help='hidden layer dimension')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)