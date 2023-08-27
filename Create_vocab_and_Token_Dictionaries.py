
# # Creates Vocabulary and Token lookup and reverse dictionary and saves in the dataset folder


import re
import pickle
from Clean_hindi_dataset import reader


vocab = set()


folderpath = "/home/dai001/Project/Dataset/"
input_filename = "clean-hindi-dataset"
output_filename = "vocab.set"
input_filepath = folderpath+input_filename
output_filepath = folderpath+output_filename


with open(output_filepath,'wb') as wp:
    with open(input_filepath,'r') as fp:

            for s in reader(fp):
                #Adding space between symbols in the sentences
                s = re.sub(r'([^\s])([!\"$%&\'()*+,-./:;=?@[\\\]_`{|}।])',r"\1 \2",s)
                s = re.sub(r'([!\"$%&\'()*+,-./:;=?@[\\\]_`{|}।])([^\s])',r"\1 \2",s)
                               
                #splitting sentences by space
                #remove anything above a length of 20
                vocab.update(set([x for x in s.split() if len(x)<20]))
                
                
                #Add <OOV> token
                vocab.add('<OOV>')
                
            print("Length of vocabulary:",len(vocab))
            pickle.dump(vocab,wp)
            

wrd2id = {}
id2wrd = {}
for i,j in enumerate(vocab,1):
    wrd2id[j] = i
    id2wrd[i] = j



with open(folderpath+"wrd2id.dictionary","wb") as w:
    pickle.dump(wrd2id,w)
    
with open(folderpath+"id2wrd.dictionary","wb") as w:
    pickle.dump(id2wrd,w)

