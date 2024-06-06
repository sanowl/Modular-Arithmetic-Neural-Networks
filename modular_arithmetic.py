import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from datasets import load_dataset, load_metric

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModularArithmeticNet(nn.Module):
    def __init__(self, p, width, task='addition'):
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
        x1 = torch.eye(self.p)[x[:, 0]]
        x2 = torch.eye(self.p)[x[:, 1]]
        x1 = torch.matmul(x1, self.U1.T)
        x2 = torch.matmul(x2, self.U2.T)
        x = torch.pow(x1 + x2, 2)
        x = torch.matmul(x, self.W.T)
        return x

def train_model(model, data_loader, epochs=1000, lr=0.005):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for batch in data_loader:
            optimizer.zero_grad()
            inputs = batch['input']
            targets = batch['target']
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
        
        if epoch % 100 == 0:
            logger.info(f'Epoch {epoch}, Loss: {loss.item()}')
    
    logger.info('Training completed.')

def evaluate_model(model, data_loader):
    model.eval()
    accuracy_metric = load_metric('accuracy')
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input']
            targets = batch['target']
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            accuracy_metric.add_batch(predictions=predicted, references=targets)
    
    accuracy = accuracy_metric.compute()
    logger.info(f'Accuracy: {accuracy["accuracy"] * 100}%')
    return accuracy["accuracy"]

def preprocess_dataset(dataset, p, task):
    inputs = []
    targets = []
    for sample in dataset:
        question = sample['question']
        parts = question.split(' ')
        n1 = int(parts[2])
        n2 = int(parts[4])
        if task == 'addition' and parts[1] == '+':
            result = (n1 + n2) % p
        elif task == 'multiplication' and parts[1] == '*':
            result = (n1 * n2) % p
        else:
            continue
        inputs.append([n1, n2])
        targets.append(result)
    return Dataset.from_dict({'input': inputs, 'target': targets})

def main():
    logger.info('Starting main function')
    # Parameters
    p = 23
    width = 128
    batch_size = 32
    epochs = 1000

    # Load and preprocess the dataset
    raw_dataset = load_dataset('math_dataset', 'arithmetic__add_or_sub')
    addition_dataset = preprocess_dataset(raw_dataset['train'], p, task='addition')
    multiplication_dataset = preprocess_dataset(raw_dataset['train'], p, task='multiplication')

    addition_data_loader = torch.utils.data.DataLoader(addition_dataset, batch_size=batch_size, shuffle=True)
    multiplication_data_loader = torch.utils.data.DataLoader(multiplication_dataset, batch_size=batch_size, shuffle=True)

    logger.info('Initializing and training model for addition')
    # Initialize and train model for addition
    addition_model = ModularArithmeticNet(p, width, task='addition')
    train_model(addition_model, addition_data_loader, epochs=epochs)
    evaluate_model(addition_model, addition_data_loader)

    logger.info('Initializing and training model for multiplication')
    # Initialize and train model for multiplication
    multiplication_model = ModularArithmeticNet(p, width, task='multiplication')
    train_model(multiplication_model, multiplication_data_loader, epochs=epochs)
    evaluate_model(multiplication_model, multiplication_data_loader)

    logger.info('Main function completed')

if __name__ == '__main__':
    main()
