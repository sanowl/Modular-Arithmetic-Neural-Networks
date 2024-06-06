import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from datasets import Dataset, load_metric
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModularArithmeticNet(nn.Module):
    def __init__(self, p: int, width: int, task: str = 'addition'):
        super(ModularArithmeticNet, self).__init__()
        self.p = p
        self.width = width
        self.task = task
        self.U1 = nn.Parameter(torch.zeros(width, p))
        self.U2 = nn.Parameter(torch.zeros(width, p))
        self.W = nn.Parameter(torch.zeros(p, width))
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            for k in range(self.width):
                if self.task == 'addition':
                    self.U1[k, :] = torch.cos(2 * np.pi / self.p * k * torch.arange(self.p))
                    self.U2[k, :] = torch.cos(2 * np.pi / self.p * k * torch.arange(self.p))
                    self.W[:, k] = torch.cos(-2 * np.pi / self.p * k * torch.arange(self.p))
                elif self.task == 'multiplication':
                    self.U1[k, :] = torch.cos(2 * np.pi / (self.p - 1) * k * torch.arange(self.p))
                    self.U2[k, :] = torch.cos(2 * np.pi / (self.p - 1) * k * torch.arange(self.p))
                    self.W[:, k] = torch.cos(-2 * np.pi / (self.p - 1) * k * torch.arange(self.p))
    
    def forward(self, x):
        x1 = torch.eye(self.p, device=x.device)[x[:, 0]]
        x2 = torch.eye(self.p, device=x.device)[x[:, 1]]
        x1 = torch.matmul(x1, self.U1.T)
        x2 = torch.matmul(x2, self.U2.T)
        x = torch.pow(x1 + x2, 2)
        x = torch.matmul(x, self.W.T)
        return x

def train_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, epochs: int = 1000, lr: float = 0.005, device: torch.device = torch.device('cpu')):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in data_loader:
            inputs = torch.tensor(batch['input'], dtype=torch.long, device=device)
            targets = torch.tensor(batch['target'], dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 100 == 0:
            logger.info(f'Epoch {epoch}, Loss: {epoch_loss / len(data_loader)}')
    
    logger.info('Training completed.')

def evaluate_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device = torch.device('cpu')):
    model.eval()
    accuracy_metric = load_metric('accuracy')
    model.to(device)
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = torch.tensor(batch['input'], dtype=torch.long, device=device)
            targets = torch.tensor(batch['target'], dtype=torch.long, device=device)
            
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            accuracy_metric.add_batch(predictions=predicted, references=targets)
    
    accuracy = accuracy_metric.compute()
    logger.info(f'Accuracy: {accuracy["accuracy"] * 100}%')
    return accuracy["accuracy"]

def generate_synthetic_dataset(num_samples: int = 1000, p: int = 23, task: str = 'addition') -> Dataset:
    inputs = []
    targets = []
    for _ in range(num_samples):
        n1 = np.random.randint(0, p)
        n2 = np.random.randint(0, p)
        if task == 'addition':
            result = (n1 + n2) % p
        elif task == 'multiplication':
            result = (n1 * n2) % p
        inputs.append([n1, n2])
        targets.append(result)
    return Dataset.from_dict({'input': inputs, 'target': targets})

def create_data_loader(dataset: Dataset, batch_size: int) -> torch.utils.data.DataLoader:
    def collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        return {key: torch.tensor([d[key] for d in batch]) for key in batch[0]}
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def main():
    logger.info('Starting main function')
    # Parameters
    p = 23
    width = 128
    num_samples = 1000
    batch_size = 32
    epochs = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate synthetic datasets
    addition_dataset = generate_synthetic_dataset(num_samples, p, task='addition')
    multiplication_dataset = generate_synthetic_dataset(num_samples, p, task='multiplication')

    # DataLoader
    addition_data_loader = create_data_loader(addition_dataset, batch_size)
    multiplication_data_loader = create_data_loader(multiplication_dataset, batch_size)

    logger.info('Initializing and training model for addition')
    # Initialize and train model for addition
    addition_model = ModularArithmeticNet(p, width, task='addition')
    train_model(addition_model, addition_data_loader, epochs=epochs, device=device)
    evaluate_model(addition_model, addition_data_loader, device=device)

    logger.info('Initializing and training model for multiplication')
    # Initialize and train model for multiplication
    multiplication_model = ModularArithmeticNet(p, width, task='multiplication')
    train_model(multiplication_model, multiplication_data_loader, epochs=epochs, device=device)
    evaluate_model(multiplication_model, multiplication_data_loader, device=device)

    logger.info('Main function completed')

if __name__ == '__main__':
    main()
