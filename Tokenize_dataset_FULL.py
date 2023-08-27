

# Tokenizes the string
#tokenize() takes in a string and returns a list of tokens
# This file when run reads the clean-hindi-dataset and tokenize the whole dataset



import pickle
import re
from Clean_hindi_dataset import reader


#loading word to id dictionary and reverse dictionary from pickle files
with open('/home/dai001/Project/Dataset/wrd2id.dictionary','rb') as wp:
    wrd2id = pickle.load(wp)
with open('/home/dai001/Project/Dataset/id2wrd.dictionary','rb') as wp:
    id2wrd = pickle.load(wp)


def tokenize(s):
    #Adding space between symbols in the sentences
    s = re.sub(r'([^\s])([!\"$%&\'()*+,-./:;=?@[\\\]_`{|}।])',r"\1 \2",s)
    s = re.sub(r'([!\"$%&\'()*+,-./:;=?@[\\\]_`{|}।])([^\s])',r"\1 \2",s)
    
    #Splitting the sentence
    return [wrd2id[i] if wrd2id.get(i)!=None else wrd2id['<OOV>'] for i in s.split()]

        
if __name__ == "__main__":
    
    #Folder to load and write data
    folderpath = "/home/dai001/Project/Dataset/"
    input_filename = "clean-hindi-dataset"
    output_filename = "Tokenized-hindi-dataset-FULL.list"
    input_filepath = folderpath+input_filename
    output_filepath = folderpath+output_filename
    
    tokens = []
    
    # reading from dataset and creating tokens
    with open(input_filepath,'r') as fp:

        for s in reader(fp):
            #checking if the sentence has spaces in between words.(Some sentence was found with no spaces in between words in dataset)
            if not any([len(x)>=20 for x in s.split()]):
                tokens.append(tokenize(s))
    


    with open(output_filepath,'wb') as wp:
            
        pickle.dump(tokens,wp)

                
  
    

