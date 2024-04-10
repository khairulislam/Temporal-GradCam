import time, os, torch
from utils.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class Experiment:
    def __init__(self, result_dir, device) -> None:
        self.result_dir = result_dir
        self.device = device
        
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
            
    @property
    def criterion(self):
        return torch.nn.CrossEntropyLoss
            
    @property
    def best_model_path(self): 
        return os.path.join(self.result_dir, 'best_model_params.pt')
    
    def train(
        self, model, train_dataloader, val_dataloader,
        epochs=25, learning_rate=1e-3
    ):
        criterion = self.criterion()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=3, factor=0.1,
            min_lr=1e-6, verbose=True
        )
        early_stopping = EarlyStopping(
            patience=5, path=self.best_model_path, 
            verbose=True
        )
        
        model.train()
        train_history = {
            'epoch':[], 'train_loss':[], 'val_loss':[]
        }
        
        for epoch in range(epochs):
            start = time.time()
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 10)
            # Each epoch has a training and validation phase
            
            running_loss = 0.0
            running_corrects, total = 0, 0

            # Iterate over data.
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                model.train()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                total += len(labels)

            if total > 0:
                running_corrects = running_corrects.double() / total

            print(f'Train loss: {running_loss:.4f} Acc: {running_corrects:.4f}')

            val_loss, val_acc = self.val(model, val_dataloader)
            print(f'Val loss: {val_loss:.4f} Acc: {val_acc:.4f}')

            lr_scheduler.step(val_loss)
            early_stopping(val_loss, model)
            
            train_history['epoch'].append(epoch+1)
            train_history['train_loss'].append(running_loss)
            train_history['val_loss'].append(val_loss)
            
            if early_stopping.early_stop:
                print('Early stopping ....\n')
                break
            print()

        time_elapsed = time.time() - start
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val loss: {early_stopping.best_score:4f}')

        # load best model weights
        print(f'Loading the best model from {self.best_model_path}')
        model.load_state_dict(torch.load(self.best_model_path))
        model.eval()
        
        return train_history
    
    def val(self, model, dataloader):
        model.eval()
        total_loss = 0.0
        running_corrects, total = 0, 0
        criterion = self.criterion()
        
        # Iterate over data.
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # statistics
            total_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
            total += len(labels)
            
        if total > 0:
            running_corrects = running_corrects.double() / total
        
        return total_loss, running_corrects
    
    def test(self, model, dataloader):
        model.eval()
        total_loss = 0.0
        criterion  = self.criterion()

        y_trues, y_preds, y_probs = [], [], []
        # Iterate over data.
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            _, preds = torch.max(outputs, 1)
            total_loss += loss.item()

            y_trues.extend(labels.detach().cpu().numpy())
            y_preds.extend(preds.detach().cpu().numpy())
            y_probs.extend(outputs[:, -1].detach().cpu().numpy())
            
        acc = accuracy_score(y_trues, y_preds)
        f1 = f1_score(y_trues, y_preds)
        auc = roc_auc_score(y_trues, y_probs)

        print(f'Loss: {total_loss:.4f}, Accuracy {acc:0.4f}, F1 {f1:0.4f}, AUC {auc:0.4f}.')
        
        return {
            'loss': total_loss,
            'acc': acc, 'f1': f1, 'auc':auc
        }
