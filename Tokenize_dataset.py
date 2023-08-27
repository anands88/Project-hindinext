
# Tokenizes the string
#tokenize() takes in a string and returns a list of tokens
# This file when run reads the clean-hindi-dataset and tokenize the whole dataset
# It outputs the tokens into pickled files part by part (min_split_value = 10) => There will be 10 or 11 parts according to the size of the tokenized list


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
    output_filename = "Tokenized-hindi-dataset.list"
    input_filepath = folderpath+input_filename
    output_filepath = folderpath+output_filename
    
    tokens = []
    
    # reading from dataset and creating tokens
    with open(input_filepath,'r') as fp:

            for s in reader(fp):
                #checking if the sentence has spaces in between words.(Some sentence was found with no spaces in between words in dataset)
                if not any([len(x)>=20 for x in s.split()]):
                    tokens.append(tokenize(s))
    
    #Splitting the tokens pickle file into parts
    min_split_value = 10
    
    part = len(tokens)//split_value
    parts = list(range(0,len(tokens),part))
    parts.append(len(tokens)-1)
    
    j = 0
    
    for i in range(1,min_split_value+2):
        with open(output_filepath+f"_{i}_of_{len(parts)-1}",'wb') as wp:
            
            #writing token file in parts
            pickle.dump(tokens[parts[j]:parts[j+1]],wp)
            j += 1
                
  
    

