# ========= 1. Imports =========
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import random
import numpy as np
import matplotlib.pyplot as plt
import time

torch.backends.cudnn.benchmark = True

# ========= 2. Espaço de Hiperparâmetros Otimizado =========
space = {
    "learning_rate": [1e-2, 5e-3, 1e-3],
    "batch_size": [32, 64, 128],
    "n_filters": [32, 64],
    "n_fc": [128, 256],
    "dropout": [0.0, 0.25, 0.5],
    "activation": ['relu', 'leakyrelu'],
    "optimizer": ["adam", "sgd"]
}

# ========= 3. Carregamento de Dados =========
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    return Subset(dataset, train_idx[:30000]), Subset(dataset, val_idx[:10000]), Subset(dataset, val_idx)

trainset, valset, full_valset = load_data()

# ========= 4. Modelo CNN com 2 Convoluções =========
class SmallCNN(nn.Module):
    def __init__(self, n_filters, n_fc, dropout, activation):
        super().__init__()
        act_dict = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU()
        }
        self.activation_fn = act_dict[activation]
        self.conv1 = nn.Conv2d(3, n_filters, 3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.num_classes = 100
        self.fc1 = None
        self.fc2 = None
        self.n_fc = n_fc

    def build(self, device):
        with torch.no_grad():
            x = torch.zeros(1, 3, 32, 32).to(device)
            x = self.pool(self.activation_fn(self.conv1(x)))
            x = self.pool(self.activation_fn(self.conv2(x)))
            x = self.dropout(x)
            flat_dim = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, self.n_fc).to(device)
        self.fc2 = nn.Linear(self.n_fc, self.num_classes).to(device)

    def forward(self, x):
        x = self.pool(self.activation_fn(self.conv1(x)))
        x = self.pool(self.activation_fn(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ========= 5. Avaliação do Modelo =========
def evaluate_model(ind, device, epochs=5):
    train_loader = DataLoader(trainset, batch_size=ind["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=ind["batch_size"], shuffle=False, num_workers=2)

    model = SmallCNN(ind["n_filters"], ind["n_fc"], ind["dropout"], ind["activation"]).to(device)
    model.build(device)
    criterion = nn.CrossEntropyLoss()

    weight_decay = 1e-4
    scheduler_step_size = 10
    scheduler_gamma = 0.5

    if ind["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=ind["learning_rate"], weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=ind["learning_rate"], momentum=0.9, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct, total = 0, 0
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            _, predicted = torch.max(out, 1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
            preds.extend(predicted.cpu().numpy())
            labels.extend(yb.cpu().numpy())

    return correct / total, np.array(preds), np.array(labels)

# ========= 6. Algoritmo Genético =========
def criar_individuo():
    return {k: random.choice(v) for k, v in space.items()}

def crossover(p1, p2, p3):
    return {k: random.choice([p1[k], p2[k], p3[k]]) for k in space}

def mutar(ind):
    k = random.choice(list(space.keys()))
    ind[k] = random.choice(space[k])
    return ind

def algoritmo_genetico(pop_size=10, geracoes=5, taxa_mutacao=0.3, device='cpu'):
    populacao = [criar_individuo() for _ in range(pop_size)]
    historico = []
    melhor_ind = None
    melhor_acc = -1
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
            pais_disponiveis = [ind for ind, acc, preds, labels in avaliacoes[:4]]
            p1, p2, p3 = random.sample(pais_disponiveis, 3)
            filho = crossover(p1, p2, p3)
            if random.random() < taxa_mutacao:
                filho = mutar(filho)
            nova_pop.append(filho)

        populacao = nova_pop

    tempo_total = time.time() - tempo_inicio
    return melhor_ind, melhor_acc, melhor_preds, melhor_labels, historico, tempo_total

# ========= 7. Visualização =========
def plot_accuracies(historico):
    plt.figure(figsize=(8, 5))
    for i in range(len(historico[0])):
        plt.plot([h[i] for h in historico], label=f"Indivíduo {i+1}")
    plt.xlabel("Geração")
    plt.ylabel("Acurácia dos melhores")
    plt.title("Evolução da acurácia")
    plt.legend()
    plt.grid()
    plt.show()

def show_stats(historico, tempo_total, melhor_ind, acc):
    print("\n========== RELATÓRIO FINAL ==========")
    print(f"Tempo total: {tempo_total:.1f} s")
    print(f"Melhor indivíduo: {melhor_ind}")
    print(f"Acurácia final: {acc:.4f}")
    acuracias = [a for h in historico for a in h]
    print(f"Média: {np.mean(acuracias):.4f} | Máx: {np.max(acuracias):.4f} | Mín: {np.min(acuracias):.4f}")

def plot_image_examples(full_valset, preds, labels, acertos=True, n=5):
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    idxs = np.where((preds == labels) if acertos else (preds != labels))[0][:n]
    if len(idxs) == 0:
        print("Nenhum exemplo encontrado.")
        return
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(idxs):
        img, label = full_valset[idx]
        img = img.permute(1,2,0) * 0.5 + 0.5
        plt.subplot(1, n, i+1)
        plt.imshow(img.numpy())
        plt.title(f"Pred:{preds[idx]}\nTrue:{labels[idx]}")
        plt.axis('off')
    plt.suptitle("Acertos" if acertos else "Erros")
    plt.show()

# ========= 8. Execução =========
device = 'cuda' if torch.cuda.is_available() else 'cpu'

melhor_ind, acc, preds, labels, historico, tempo_total = algoritmo_genetico(
    pop_size=10, geracoes=10, taxa_mutacao=0.3, device=device
)

show_stats(historico, tempo_total, melhor_ind, acc)
plot_accuracies(historico)

print("\n5 exemplos ACERTADOS:")
plot_image_examples(full_valset, preds, labels, acertos=True, n=5)

print("\n5 exemplos ERRADOS:")
plot_image_examples(full_valset, preds, labels, acertos=False, n=5)