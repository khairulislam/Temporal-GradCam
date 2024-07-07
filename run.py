import os, torch, random, json, gc
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
            SEED = args.seed = experiment_seeds[args.itr_num-1]
            print(f'{args.itr_num}-th iteration using seed {SEED}.')
            
    else:
        # running an iteration from the whole experiment
        SEED = args.seed = itr_seed
        print(f'{args.itr_num}-th iteration using seed {SEED}.')
        seed_everything(SEED)
    
    # set result directory and call iterations
    if args.itr_num is not None:
        assert args.itrs is not None and args.itr_num <= args.itrs, 'itr_num must be smaller than itrs'
        result_dir = os.path.join('OASIS_2D', args.result_dir, args.model, str(args.itr_num))
        
    elif args.itrs is not None:
        for itr_num in range(1, args.itrs+1):
            args.itr_num = itr_num
            # args.seed = int(experiment_seeds[itr_num-1])
            main(args, int(experiment_seeds[itr_num-1]))
        return
    else:
        result_dir = os.path.join('OASIS_2D', args.result_dir, args.model)
    
    # Get cpu, gpu device for training.
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    # Get pretrained model
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    model = model_dict[args.model](weights='DEFAULT')

    # Here, we need to freeze all the network except the final layer. 
    # We need to set requires_grad = False to freeze the parameters 
    # so that the gradients are not computed in backward().
    for param in model.parameters():
        param.requires_grad = False
        
    if args.model == 'ViT':
        num_ftrs = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(num_ftrs, 2)
        transform = vit_transforms
    elif args.model == 'ResNet':
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        transform = None
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented !")

    model = model.to(device)
    
    # Prepare dataset and dataloader
    train_dataset = OASIS_Dataset(
        flag='train', seed=SEED, 
        transform=transform,
        vit=args.model=='ViT'
    )
    test_dataset = OASIS_Dataset(
        flag='test', seed=SEED, 
        transform=transform, 
        vit=args.model=='ViT'
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
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
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)