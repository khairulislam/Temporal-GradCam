import os, torch, random, json
import numpy as np
import matplotlib.pyplot as plt 
from torchvision import transforms, models
from torch.utils.data import DataLoader
import argparse
from OASIS_2D.dataset import OASIS_Dataset
from OASIS_2D.oasis import Experiment
import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

data_transforms = {
    'train': transforms.Compose([
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
        # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        transforms.ToTensor(), 
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(), # PIL Image or numpy.ndarray (H x W x C)
        # transforms.Resize(256),
        # transforms.CenterCrop(224)
    ]),
}

vit_transforms = {
    'train': transforms.Compose([
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
        # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        transforms.ToTensor(), 
        # transforms.RandomResizedCrop(224),
        transforms.Resize(224),
        # transforms.RandomHorizontalFlip(),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(), # PIL Image or numpy.ndarray (H x W x C)
        transforms.Resize(224),
        # transforms.CenterCrop(224)
    ]),
}

def main(args):
    SEED = args.seed
    print(f'Random seed {SEED}')
    seed_everything(SEED)
    
    # Get cpu, gpu device for training.
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    # Prepare dataset and dataloader
    train_dataset = OASIS_Dataset(flag='train', seed=SEED)
    test_dataset = OASIS_Dataset(flag='test', seed=SEED)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Get pretrained model
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    model = getattr(models, args.model)(weights='DEFAULT')

    # Here, we need to freeze all the network except the final layer. 
    # We need to set requires_grad = False to freeze the parameters 
    # so that the gradients are not computed in backward().
    for param in model.parameters():
        param.requires_grad = False
        
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model.fc = torch.nn.Linear(num_ftrs, 2)

    model = model.to(device)
    
    if args.itr is not None:
        result_dir = os.path.join('OASIS_2D', args.result_dir, args.model, str(args.itr))
    else:
        result_dir = os.path.join('OASIS_2D', args.result_dir, args.model)
    # result_dir = os.path.join('OASIS_2D', 'results', 'ResNet18')
    exp = Experiment(result_dir=result_dir, device=device)

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

def get_parser():
    parser = argparse.ArgumentParser(
        description='Train model', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--itr', default=None, type=int, help='iteration number')
    # basic config
    parser.add_argument('--model', default='resnet18', type=str, help='model name', choices=['resnet18'])
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    parser.add_argument('--result_dir', default='results',  type=str, help='directory to save the result')
    parser.add_argument('--test', action='store_true', help='load a pretrained model for testing a split')
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    
    # model arguments
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)