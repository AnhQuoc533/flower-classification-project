import torch
from torch import nn, optim
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt

class VGGModel:

    def __init__(self, arch: str = None, neurons: list[int] = None, drop_p=.0, file: str = None):
        # Load model's information from file
        if file is not None:
            checkpoint = torch.load(file)
            arch = checkpoint['arch']
            neurons = checkpoint['neurons']
            drop_p = checkpoint['drop_p']
            self.idx_to_cls = checkpoint['idx_to_cls']
        elif arch is None or neurons is None:
            raise ValueError('arch and neurons are expected if model is not loaded from file.')

        # Get model architecture
        if arch == 'vgg11':
            self.model = models.vgg11(weights='DEFAULT')
        elif arch == 'vgg13':
            self.model = models.vgg13(weights='DEFAULT')
        elif arch == 'vgg16':
            self.model = models.vgg16(weights='DEFAULT')
        elif arch == 'vgg19':
            self.model = models.vgg19(weights='DEFAULT')
        else:
            raise ValueError("arch must be one of 'vgg11', 'vgg13', 'vgg16' or 'vgg19'.")

        self.arch = arch
        self.drop_p = drop_p

        # Freeze out parameters of feature part
        for param in self.model.parameters():
            param.requires_grad = False

        # Build classification part
        classifier_layers = [nn.Linear(self.model.classifier[0].in_features, neurons[0])]
        for h1, h2 in zip(neurons, neurons[1:]):
            classifier_layers += [nn.ReLU(), nn.Dropout(self.drop_p), nn.Linear(h1, h2)]
        self.model.classifier = nn.Sequential(*classifier_layers)

        # Load state_dict to the model from checkpoint file
        if file is not None:
            self.model.classifier.load_state_dict(checkpoint['state_dict'])

    def train(
        self, 
        train_data, 
        val_data, 
        lr: float, 
        epochs: int, 
        batch_size: int = 32, 
        gpu=False, 
        plot_loss=False
    ):

        # Create DataLoader for training and validation data
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
        # Save the mapping from index to class
        self.idx_to_cls = {value: key for key, value in train_data.class_to_idx.items()}

        # Utilize gpu
        device = torch.device('cpu')
        if gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.model.to(device)
            else:
                print('CUDA is not available in your device. CPU will be utilized instead.')

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=lr)
        train_costs, val_costs = [], [None]

        # Train model
        for i in range(epochs):
            sum_loss = 0
            for X, y in trainloader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()

                logits = self.model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()

            # Print and save training costs
            train_costs.append(sum_loss / len(trainloader.dataset))
            print(f"Epoch: {i+1:3} -- Training cost (before epoch): {train_costs[-1]:.8f} -- ", end='')

            # Compute and save validation costs
            with torch.no_grad():
                self.model.eval()
                
                sum_loss = accuracy = 0
                for X, y in valloader:
                    X, y = X.to(device), y.to(device)

                    # Compute validation cost
                    logits = self.model.forward(X)
                    sum_loss += criterion(logits, y).item()

                    # Compute validation accuracy
                    y_hat = F.softmax(logits, dim=1)
                    predictions = y_hat.argmax(dim=1)
                    accuracy += (predictions == y).count_nonzero()

                val_costs.append(sum_loss / len(valloader.dataset))
                print(f"Validation cost (after epoch): {val_costs[-1]:.8f} -- ", end='')
                print(f"Validation accuracy: {accuracy.item()/len(valloader.dataset):.8f}")

                self.model.train()

        # Plot training and validation costs
        if plot_loss:
            plt.plot(train_costs, label='Training cost')
            plt.plot(val_costs, label='Validation cost')

            plt.xticks([i for i in range(epochs + 1)])
            plt.xlabel('Epoch')
            plt.ylabel('Cost')
            plt.legend()

            plt.show()

    def save(self, filename: str, dir: str = ''):
        checkpoint = {
            'arch': self.arch,
            'neurons': [layer.out_features for layer in self.model.classifier if type(layer) is nn.Linear],
            'drop_p': self.drop_p,
            'state_dict': self.model.classifier.state_dict(),
            'idx_to_cls': self.idx_to_cls
        }

        if dir != '':
            dir = dir if dir[-1] in "/\\" else dir + '/'
        torch.save(checkpoint, dir + filename)
        
        print("Model's information is saved at", dir + filename)

    def predict(self, X: torch.Tensor, topk=1, gpu=False):
        # Utilize gpu
        device = torch.device('cpu')
        if gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.model.to(device)
            else:
                print('CUDA is not available in your device. CPU will be utilized instead.')
                
        if X.ndim != 3:
            raise ValueError('3D Tenser expected for X.')
        else:
            X = X.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            self.model.eval()
            y_hat = F.softmax(self.model.forward(X), dim=1)
            top_values, top_idx = y_hat.topk(topk, dim=1)
            self.model.train()

        # Convert neuron indices to classes
        top_classes = [self.idx_to_cls[idx] for idx in top_idx[0].tolist()]

        return top_values[0].tolist(), top_classes
