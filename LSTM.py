import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np


class TextDataset(Dataset):
    """Dataloader Structure for classification"""
    def __init__(self, text_data, labels_data):
        super().__init__()
        self.text_data = text_data
        self.labels_data = labels_data

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        return self.text_data[idx], self.labels_data[idx]


class LSTMClassifier(nn.Module):
    """Long short-term memory model"""
    def __init__(self, embeddings, hidden_size, num_layers, dropout_prob=0.4):
        """neural network structure"""
        super().__init__()
        # input layer
        self.embedding_layer = self.load_pretrained_embeddings(embeddings)
        # LSTM layer
        self.lstm = nn.LSTM(embeddings.shape[1], hidden_size, num_layers, dropout=dropout_prob, batch_first=True)
        # activation layer
        self.non_linearity = nn.ReLU()
        # output layer
        self.clf = nn.Linear(hidden_size, embeddings.shape[0])

    def load_pretrained_embeddings(self, embeddings):
        """load glove embeddings as input"""
        embedding_layer = nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        embedding_layer.weight.data = torch.Tensor(embeddings).float()
        return embedding_layer

    def forward(self, inputs):
        """forward propagation"""
        logits = self.embedding_layer(inputs)
        logits = self.lstm(logits)[0][:, -1, :]
        logits = self.non_linearity(logits)
        logits = self.clf(logits)
        return logits


def train(device, app, file_name, window_length=3, hidden_size=600, num_layers=2, dropout_prob=0.4, batch_size=64,
          start=0, num_epochs=1000, patience=10, model=None):
    """
    train neural network model
    :param device: current device
    :param app: vocabulary
    :param file_name: path to training corpus file
    :param window_length: window length of input
    :param hidden_size: number of hidden neurons in each layer 
    :param num_layers: number of hidden layer
    :param dropout_prob: probability of dropout
    :param batch_size: number of data for each training batch
    :param start: start epochs (for additional training)
    :param num_epochs: total round of training
    :param patience: patience for early end
    :param model: previous trained model (for additional training)
    :return: trained model
    """
    print("training")
    if not model:
        # TODO
        model = LSTMClassifier(app.embeddings, hidden_size, num_layers, dropout_prob)
        model.to(device)
    # TODO
    criterion = nn.CrossEntropyLoss()
    # TODO
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # print(model)

    train_data, train_label = app.load_data(file_name, window_length)
    train_loader = torch.utils.data.DataLoader(dataset=TextDataset(train_data, train_label),
                                               batch_size=batch_size,
                                               shuffle=True)

    losses = []
    for epoch in range(start, num_epochs):
        temp = []
        print("*" * 40 + str(epoch) + "*" * 40)
        model.train()  # this enables regularization, which we don't currently have
        for i, (data_batch, batch_labels) in enumerate(train_loader):
            preds = model(data_batch.to(device))
            loss = criterion(preds, batch_labels.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            temp.append(loss.item())
        cur_loss = sum(temp) / len(temp)
        losses.append(cur_loss)
        print(cur_loss)
        if len(losses) >= patience and cur_loss > max(losses[-1 - patience:-1]):
            torch.save(model, "data/model_%d.pt" % epoch)
            break
        if epoch % patience == 0:
            torch.save(model, "data/model_%d.pt" % epoch)
    return model


def test(device, app, model, file_name, window_length=3, batch_size=64):
    """
    testing neural network model
    :param device: current device
    :param app: vocabulary
    :param model: trained model
    :param file_name: path to testing corpus file
    :param window_length: window length of input
    :param batch_size: number of data for each test batch
    :return: accuracy
    """
    print("testing")
    test_data, test_label = app.load_data(file_name, window_length)
    remove = []
    for i in range(len(test_label)):
        if test_label[i] < 2:
            remove.append(i)
    test_data = np.delete(test_data, remove, axis=0)
    test_label = np.delete(test_label, remove, axis=0)
    test_loader = torch.utils.data.DataLoader(dataset=TextDataset(test_data, test_label),
                                              batch_size=batch_size,
                                              shuffle=True)

    print(evaluate(model, test_loader, device))


def evaluate(model, dataloader, device):
    """evaluate model"""
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for batch_text, batch_labels in dataloader:
            preds = model(batch_text.to(device))
            all_preds.append(np.argmax(preds.detach().cpu().numpy(), axis=1))
            all_labels.append(batch_labels.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    accuracy = (all_labels == all_preds).mean()
    return accuracy


if __name__ == '__main__':
    from Vocabulary import Vocabulary
    from pickle import load

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    with open("data/vocab", "rb") as f:
        app = load(f)

    # TODO
    window_length = 4
    train_file = "corpus/train_business.pkl"
    model = torch.load("data/model_bus_l.pt", map_location=torch.device('cpu'))
    # model = train(device, app, train_file, window_length)

    test_file = "corpus/test_all.pkl"
    test(device, app, model, test_file, window_length)
    test_file = "corpus/test_business.pkl"
    test(device, app, model, test_file, window_length)
    test_file = "corpus/test_enter.pkl"
    test(device, app, model, test_file, window_length)
    test_file = "corpus/test_pol.pkl"
    test(device, app, model, test_file, window_length)
    test_file = "corpus/test_sport.pkl"
    test(device, app, model, test_file, window_length)
    test_file = "corpus/test_tech.pkl"
    test(device, app, model, test_file, window_length)
