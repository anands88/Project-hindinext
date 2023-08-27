
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from indicnlp.tokenize import indic_tokenize
import fasttext
import matplotlib.pyplot as plt
from Clean_hindi_dataset import reader
import random
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
print("Imported libraries...")


###-----------------------------------
### Constants
###-----------------------------------

EPOCHS = 10
LR = 0.01
DATASET_PATH = "/home/dai001/Project/Dataset/clean-hindi-dataset"
SAVE_FOLDER = "/home/dai001/Project/graphs/"
MODEL_PATH = "/home/dai001/Project/model/transformer-model-best.pth"
MODEL_STATE_PATH = "/home/dai001/Project/model/transformer-model-best-state.pth"
BATCH_SIZE = 32


# parameters for Matplotlib
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'
         }

CMAP = plt.cm.coolwarm

plt.rcParams.update(params)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Working on : ",device)


###-----------------------------------
### Positional Encoding class
###-----------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


###-----------------------------------
### Transformer Class
###-----------------------------------

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,max_seq_len):
        super(TransformerModel, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.dense = nn.Linear(d_model, embedding_matrix.size(0))
        
    def forward(self, src, tgt):
        #check if need for changing order of embedding shape
        #src_embedding = src.permute(1, 0, 2)  # Permute to (seq_length, batch_size, embedding_dim)
        #tgt_embedding = tgt.permute(1, 0, 2)
        
        src_embedding = self.positional_encoding(src)
        tgt_embedding = self.positional_encoding(tgt)
        
        encoder_output = self.encoder(src_embedding)
        decoder_output = self.decoder(tgt_embedding, encoder_output)
        
        #check for need if order is changed in previous step
        #output = output.permute(1, 0, 2)  # Permute back to (batch_size, seq_length, embedding_dim)
        output = self.dense(decoder_output)
        
        return output



###-----------------------------------
### Embedding Class
###-----------------------------------

class Embedding_model(nn.Module):
    def __init__(self, embedding_matrix):
        super(Embedding_model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        
    def forward(self, src, tgt):
        src_embedding = self.embedding(src)
        tgt_embedding = self.embedding(tgt)
        
        return src_embedding, tgt_embedding



# Load pre-trained FastText Hindi embeddings
embedding_loaded_model = fasttext.load_model("cc.hi.300.bin")

#Create the embedding matrix which contains all the embedding vectors for the entire vocabulary
#Embedding matrix has each words in rows and its embeddings as columns(this model has 300 columns)
embedding_matrix = torch.tensor(np.array([embedding_loaded_model.get_word_vector(word) for word in embedding_loaded_model.words]))

# Load pre-trained tokenizer
tokenizer = indic_tokenize


#specify all parameters which goes into the transformer model
d_model = embedding_matrix.size(1)
nhead = 10
num_encoder_layers = 6
num_decoder_layers = 6
max_seq_len = 100  


#Initializing the model
model = TransformerModel(d_model, nhead, num_encoder_layers, num_decoder_layers,max_seq_len).to(device)
print("MODEL:")
print(model)


embed = Embedding_model(embedding_matrix)
print("Embedding model:")
print(embed)


###-----------------------------------
### Custom Dataset Class
###-----------------------------------

class DS(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_length
        self.data = []

        with open(file_path, 'r', encoding='utf-8') as fp:
            for s in reader(fp):
                if s:
                    self.data.append(s)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        text = self.data[idx]
        tokens = self.tokenizer.trivial_tokenize(text)[:self.max_seq_len]
        
        # Taking random size of words from the text to train instead of the entire sentence
        if len(tokens)>3:
            random_size = random.randint(3,len(tokens))
            random_start = random.randint(0,len(tokens)-1)
            selected = np.array(tokens[random_start:random_start+random_size])
            target = selected[-1]
            features = selected[:-1]
        else:
            features = np.array(tokens[:-1])
            target = tokens[-1]
        # Converting into tensors    
        features = torch.tensor(features)
        target = torch.tensor(target)

        return features,target 


#Dataset
train_ds = DS(DATASET_PATH+"-train", tokenizer, max_seq_len)
test_ds = DS(DATASET_PATH+"-test", tokenizer, max_seq_len)


#Dataloader
train_dl = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
test_dl = DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=True)


###-----------------------------------
### Training Model
###-----------------------------------

print("Beginning Training...")
# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# Training loop
train_losses = []
test_losses = []
train_accs = []
test_accs = []
best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for features, target in train_dl:
        #Embedding features and target
        features, target = embed(features, target)
        
        #Sending features and target to gpu
        features, target = features.to(device), target.to(device)

        optimizer.zero_grad()
        
        #Forward pass through model
        output = model(features, target)  

        # Reshape the output and target tensors to calculate loss
        output = output.view(-1, embedding_matrix.size(0))
        target = target.view(-1)

        loss = loss_fn(output, target)
        acc = accuracy_score(target.cpu().numpy(),output.cpu().numpy())
        loss.backward()
        optimizer.step()
        

        total_loss += loss.item()
        total_acc += acc

    avg_train_loss = total_loss / len(train_dl)
    avg_train_acc = total_acc / len(train_dl)
    
    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)
    

    # Testing loop
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for features, target in test_dl:
            features, target = features.to(device), target.to(device)

            output = model(features, features)  # Auto-regressive prediction

            output = output.view(-1, embedding_matrix.size(0))
            target = target.view(-1)

            loss = criterion(output, target)
            acc = accuracy_score(target.cpu().numpy(),output.cpu().numpy())
            
            total_test_loss += loss.item()
            total_test_acc += acc()

    avg_test_loss = total_test_loss / len(test_dl)
    avg_test_acc = total_test_acc / len(test_dl)
    
    if avg_test_acc > best_acc:
        torch.save(model,MODEL_PATH)
        torch.save(model.state_dict(),MODEL_STATE_PATH)
    
    test_losses.append(avg_test_loss)
    test_accs.append(avg_test_acc)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f} - Train Acc: {avg_train_acc:.4f} - Test Acc: {avg_test_acc:.4f} ")


###-----------------------------------
### Function to plot Loss Curve
###-----------------------------------


def plot_hist(hist_df, save_folder):
    # instantiate figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # properties  matplotlib.patch.Patch
    props = dict(boxstyle='round', facecolor='cyan', alpha=0.5)

    # Where was min loss
    best = hist_df[hist_df['test_loss'] == hist_df['test_loss'].min()]

    # pick first axis
    ax = axes[0]

    # Plot all losses
    hist_df.plot(x='epoch', y=['loss', 'test_loss'], ax=ax)

    # little beautification
    txtFmt = "Loss: \n  train: {:6.4f}\n   test: {:6.4f}"
    txtstr = txtFmt.format(hist_df.iloc[-1]['loss'],
                           hist_df.iloc[-1]['test_loss'])  # text to plot

    # place a text box in upper middle in axes coords
    ax.text(0.3, 0.95, txtstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    # Mark arrow at lowest
    ax.annotate(f'Min: {best["test_loss"].to_numpy()[0]:6.4f}',  # text to print
                xy=(best['epoch'].to_numpy(), best["test_loss"].to_numpy()[0]),  # Arrow start
                xytext=(best['epoch'].to_numpy() + 0.01, best["test_loss"].to_numpy()[0] + 0.01),  # location of text
                fontsize=14, va='bottom', ha='right', bbox=props,  # beautification of text
                arrowprops=dict(facecolor='cyan', shrink=0.05))  # arrow

    # Draw vertical line at best value
    ax.axvline(x=best['epoch'].to_numpy(), color='green', linestyle='-.', lw=3)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title('Errors')
    ax.grid()
    ax.legend(loc='upper left')  # model legend to upper left

    # pick second axis
    ax = axes[1]

    # Plot accuracies
    hist_df.plot(x='epoch', y=['acc', 'test_acc'], ax=ax)

    # little beautification
    txtFmt = "Accuracy: \n  train: {:6.4f}\n  test:  {:6.4f}"
    txtstr = txtFmt.format(hist_df.iloc[-1]['acc'],
                           hist_df.iloc[-1]['test_acc'])  # text to plot

    # place a text box in lower middle in axes coords
    ax.text(0.3, 0.2, txtstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # Mark arrow at lowest
    ax.annotate(f'Best: {best["test_acc"].to_numpy()[0]:6.4f}',  # text to print
                xy=(best['epoch'].to_numpy(), best["test_acc"].to_numpy()[0]),  # Arrow start
                xytext=(best['epoch'].to_numpy() - 2, best["test_acc"].to_numpy()[0]),  # location of text
                fontsize=14, va='bottom', ha='right', bbox=props,  # beautification of text
                arrowprops=dict(facecolor='cyan', shrink=0.05))  # arrow

    # Draw a vertical line at best value
    ax.axvline(x=best['epoch'].to_numpy(),
               color='green',
               linestyle='-.', lw=3)

    # Labels
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title('Accuracies')
    ax.grid()
    ax.legend(loc='lower left')

    # Save the figure instead of displaying
    save_path = os.path.join(SAVE_FOLDER, "loss_and_accuracy_plot.png")
    plt.savefig(save_path)

    # Close the figure to free up resources
    plt.close(fig)


#plotting the loss and accuracy
print("Plotting the loss and accuracy curves...")
loss_df = pd.DataFrame({'epoch' : n_epoch, 'loss' : loss, 'test_loss': tloss, 'acc' : acc, 'test_acc': tacc})
plot_hist(loss_df)

