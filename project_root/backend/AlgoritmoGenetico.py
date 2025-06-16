#exemplo do gemini que estamos testando
# genetic_algorithm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import random, numpy as np, time
import os # Importar os para criar diretório para dados

# Garante que o diretório 'data' exista para o CIFAR10
if not os.path.exists('./data'):
    os.makedirs('./data')

torch.backends.cudnn.benchmark = True

# ========= 1. Espaço de Hiperparâmetros =========
space = {
    "learning_rate": [1e-2, 5e-3, 1e-3, 5e-4],
    "batch_size": [8, 16, 32],
    "n_filters": [16, 32, 64],
    "n_fc": [64, 128, 256],
    "dropout": [0.0, 0.25, 0.5]
}

# ========= 2. Carregamento de Dados =========
# Variáveis globais para os datasets
trainset, valset, full_valset = None, None, None

def load_data_once():
    """Carrega os dados apenas uma vez para evitar downloads repetidos."""
    global trainset, valset, full_valset
    if trainset is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        # Usando o dataset completo para treino e validação, se você quiser mais dados.
        # Caso contrário, mantenha o slicing para testes rápidos.
        train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
        trainset = Subset(dataset, train_idx) # Removed [:4000]
        valset = Subset(dataset, val_idx)     # Removed [:1000]
        full_valset = Subset(dataset, val_idx) # This is redundant if valset uses full val_idx

# Chame esta função na inicialização do backend
# load_data_once() # Será chamado no app.py

# ========= 3. CNN =========
class SmallCNN(nn.Module):
    def __init__(self, n_filters, n_fc, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(3, n_filters, 3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(n_filters * 2, n_filters * 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = self.fc2 = None
        self.n_fc = n_fc

    def build(self, device):
        with torch.no_grad():
            x = torch.zeros(1, 3, 32, 32).to(device)
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = self.dropout(x)
            flatten_dim = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flatten_dim, self.n_fc).to(device)
        self.fc2 = nn.Linear(self.n_fc, 10).to(device)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ========= 4. Avaliação =========
def evaluate_model(ind, device, epochs=5):
    # Assegurar que os datasets globais estão carregados
    load_data_once()

    train_loader = DataLoader(trainset, batch_size=ind["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=ind["batch_size"], shuffle=False, num_workers=2)

    model = SmallCNN(ind["n_filters"], ind["n_fc"], ind["dropout"]).to(device)
    model.build(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=ind["learning_rate"])

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    preds_list, labels_list = [], [] # Mudei o nome para evitar conflito com 'labels' do evaluate_model
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            _, predicted = torch.max(out, 1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
            preds_list.extend(predicted.cpu().numpy())
            labels_list.extend(yb.cpu().numpy())

    return correct / total, np.array(preds_list), np.array(labels_list)


# ========= 5. Algoritmo Genético Otimizado =========
def criar_individuo():
    return {k: random.choice(v) for k, v in space.items()}

def crossover(p1, p2):
    return {k: random.choice([p1[k], p2[k]]) for k in space}

def mutar(ind):
    k = random.choice(list(space.keys()))
    ind[k] = random.choice(space[k])
    return ind

def run_genetic_algorithm(pop_size=6, geracoes=10, taxa_mutacao=0.3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando device: {device}")

    populacao = [criar_individuo() for _ in range(pop_size)]
    historico = []
    melhor_ind = None
    melhor_acc = -1
    melhor_preds = None
    melhor_labels = None
    tempo_inicio = time.time()

    for g in range(geracoes):
        print(f"\n--- Geração {g+1} ---")
        avaliacoes = [(ind, *evaluate_model(ind, device)) for ind in populacao]
        avaliacoes.sort(key=lambda x: x[1], reverse=True)
        historico.append([a[1] for a in avaliacoes[:4]])

        if avaliacoes[0][1] > melhor_acc:
            melhor_ind, melhor_acc, melhor_preds, melhor_labels = avaliacoes[0]

        print(f"Melhor da geração {g+1}: {avaliacoes[0][1]:.4f} | {avaliacoes[0][0]}")

        nova_pop = [ind for ind, acc, preds, labels in avaliacoes[:2]]
        while len(nova_pop) < pop_size:
            p1, p2 = random.sample(nova_pop, 2)
            filho = crossover(p1, p2)
            if random.random() < taxa_mutacao:
                filho = mutar(filho)
            nova_pop.append(filho)
        populacao = nova_pop

    tempo_total = time.time() - tempo_inicio
    return melhor_ind, melhor_acc, melhor_preds.tolist(), melhor_labels.tolist(), historico, tempo_total