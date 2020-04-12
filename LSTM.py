import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np


class TextDataset(Dataset):
    def __init__(self, text_data, labels_data):
        super().__init__()
        self.text_data = text_data
        self.labels_data = labels_data

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        return self.text_data[idx], self.labels_data[idx]


class LSTMClassifier(nn.Module):

    def __init__(self, embeddings, hidden_size, num_layers, dropout_prob=0.3):
        super().__init__()
        self.embedding_layer = self.load_pretrained_embeddings(embeddings)
        self.lstm = nn.LSTM(embeddings.shape[1], hidden_size, num_layers, dropout=dropout_prob, batch_first=True)
        self.non_linearity = nn.ReLU()
        self.clf = nn.Linear(hidden_size, embeddings.shape[0])

    def load_pretrained_embeddings(self, embeddings):
        embedding_layer = nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        embedding_layer.weight.data = torch.Tensor(embeddings).float()
        return embedding_layer

    def forward(self, inputs):
        logits = self.embedding_layer(inputs)
        logits = self.lstm(logits)[0][:, -1, :]
        logits = self.non_linearity(logits)
        logits = self.clf(logits)
        return logits


def evaluate(model, dataloader, device):
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


def train(device, app, file_name, window_length=3, hidden_size=500, num_layers=2, BATCH_SIZE=64, NUM_EPOCHS=1000,
          patience=3):
    model = LSTMClassifier(app.embeddings, hidden_size, num_layers)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # print(model)

    train_data, train_label = app.load_data(file_name, window_length)
    train_loader = torch.utils.data.DataLoader(dataset=TextDataset(train_data, train_label),
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    losses = []
    for epoch in range(NUM_EPOCHS):
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
        if cur_loss > max(losses[-1 - patience:]):
            torch.save(model, "model_%d.pt" % epoch)
            break
        if epoch % 10 == 0:
            torch.save(model, "model_%d.pt" % epoch)


def test(device, app, model, file_name, window_length=3, BATCH_SIZE=64):
    test_data, test_label = app.load_data(file_name, window_length)
    test_loader = torch.utils.data.DataLoader(dataset=TextDataset(test_data, test_label),
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    print(evaluate(model, test_loader, device))


if __name__ == '__main__':
    from Vocabulary import Vocabulary
    from pickle import load

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    with open("vocab", "rb") as f:
        app = load(f)

    window_length = 3
    train(device, app, "small_train.txt", window_length)

    model = torch.load("model_0.pt")
    test(device, app, model, "small_test.txt")

    top_k = 3
    example = ["open", "up", "the"]
    for _ in range(10):
        print(example)
        idx = app.get_idx(example[-window_length:])

        test = torch.from_numpy(np.array([idx]))
        results = model(test.to(device)).detach().cpu().numpy()

        best = None
        for res in results:
            temp = res.argsort()[::-1]
            for i in temp[:top_k]:
                print(i, app.get_word(i), end="\t")
            print()
            best = app.get_word(temp[0])
        example.append(best)
