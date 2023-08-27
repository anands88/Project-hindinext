

from Clean_hindi_dataset import reader


#Folder to load and write data
folderpath = "/home/dai001/Project/Dataset/"
input_filename = "clean-hindi-dataset"
output_filename = "clean-hindi-dataset"
input_filepath = folderpath+input_filename
output_filepath = folderpath+output_filename


counter = 0
i = 0
postfix = ['train','valid','test']


with open(input_filepath,'r') as fp:
    
        
        for s in reader(fp):
            #checking if the sentence has spaces in between words.(Some sentence was found with no spaces in between words in dataset)
            if not any([len(x)>=20 for x in s.split()]):
                with open(output_filepath+f"-{postfix[i]}",'a') as wp:
                    wp.write(s+"\n\n")
                counter += 1
                
            if counter > 80000000:
                if counter > 90000000:
                    i = 2
                else:
                    i = 1
                

