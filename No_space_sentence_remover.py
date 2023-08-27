
# Removes sentences that doesn't have spaces in between words from the clean-hindi-dataset


from Clean_hindi_dataset import reader


#Folder to load and write data
folderpath = "/home/dai001/Project/Dataset/"
input_filename = "clean-hindi-dataset"
output_filename = "new-clean-hindi-dataset"
input_filepath = folderpath+input_filename
output_filepath = folderpath+output_filename


with open(input_filepath,'r') as fp:
    with open(output_filepath,'a') as wp:
        
        for s in reader(fp):
            #checking if the sentence has spaces in between words.(Some sentence was found with no spaces in between words in dataset)
            if not any([len(x)>=20 for x in s.split()]):
                wp.write(s+"\n\n")

