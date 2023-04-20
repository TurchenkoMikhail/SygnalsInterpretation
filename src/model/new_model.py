import torch
from torch.utils.data import Dataset
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):

  def __init__(self, texts, targets, tokenizer, max_len=512):
    self.texts = texts
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts[idx])
    target = self.targets[idx]

    encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


class BertClassifier:

    def __init__(self, model_path, tokenizer_path, n_classes=2, epochs=1, model_save_path='bert.pt'):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.max_len = 512
        self.epochs = epochs
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes)
        self.model.to(self.device)

    def preparation(self, X_train, y_train, X_valid, y_valid):
        # create datasets
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=2, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=2, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0

        for data in self.train_loader:
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            loss = self.loss_fn(outputs.logits, targets)

            correct_predictions += torch.sum(preds == targets)

            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        train_acc = correct_predictions.double() / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss

    def eval(self):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        val_acc = correct_predictions.double() / len(self.valid_set)
        val_loss = np.mean(losses)
        return val_acc, val_loss

    def train(self):
        best_accuracy = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_acc, train_loss = self.fit()
            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval()
            print(f'Val loss {val_loss} accuracy {val_acc}')
            print('-' * 10)

            #if val_acc > best_accuracy:
            torch.save(self.model, './bert.pt')
            best_accuracy = val_acc

        self.model = torch.load('./bert.pt')

    def predict(self, text):
        self.model = torch.load('./bert.pt')
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )
        tmp = torch.argmax(outputs.logits, dim=1)
        tmp = tmp.cpu()
        tmp = tmp.numpy()
        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return prediction


def compile():
    classifier = BertClassifier(
            model_path='cointegrated/rubert-tiny',
            tokenizer_path='cointegrated/rubert-tiny',
            n_classes=2,
            epochs=2,
            model_save_path='bert.pt'
    )
    VOCABULARY_PATH="dataset/fillers.txt"
    POSITIVE_PATH="dataset/pos/text1.txt"
    NEGATIVE_PATH="dataset/neg/text1.txt"


    def read_file(file_path: str):
        lines = []
        with open(file_path, mode="r", encoding="utf-8") as file:
            for line in file:
                lines.append(line.rstrip())
        return lines

    vocabulary = read_file(VOCABULARY_PATH)
    positive_data = read_file(POSITIVE_PATH)
    pos_labels = [1 for i in range(len(positive_data))] # without fillers
    negative_data = read_file(NEGATIVE_PATH)
    neg_labels = [0 for i in range(len(negative_data))] # with fillers

    X = positive_data + negative_data
    Y = pos_labels + neg_labels


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=0)
    train_data = X_train
    test_data  = X_test

    classifier.preparation(
            X_train=train_data,
            y_train=y_train,
            X_valid=test_data,
            y_valid=y_test
        )
    classifier.train()

    texts = list(X_valid)
    labels = list(y_valid)

    predictions = [classifier.predict(t) for t in texts]

    my_tests = ["эта машина кажется короче красного грузовика",
                "короче, я пришел на работу и увидел директора",
                "Я увидел на дереве змею, которая короче моей кошки",
                "Мне нужна машина современного типа для езды по городу",
                "И я ему отвечаю что, типа, он должен поехать со мной"
                ]

    for test in my_tests:
        pred = classifier.predict(test)
        if pred == 0:
            ans = "filler"
        else:
            ans = "not filler"
        print(f"Test: {test} \nPrediction: {pred} \nans: {ans}")

    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1score = precision_recall_fscore_support(labels, predictions, average='macro')[:3]

    print(f'precision: {precision}, recall: {recall}, f1score: {f1score}')


compile()