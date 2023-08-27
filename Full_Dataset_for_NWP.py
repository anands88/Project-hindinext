
import ast


token_path = '/home/dai001/Project/Dataset/Tokenized-hindi-dataset'
folderpath = "/home/dai001/Project/Dataset/Full-dataset/"
output_filename = "Full-dataset-NWP"
output_filepath = folderpath+output_filename


# Creating data for Next word prediction
line_counter = 0
file_counter = 1
with open(token_path,'r') as r:
    
    for line in r:
        line_counter += 1
        #change string "[1,2,3]" to list [1,2,3]
        line = ast.literal_eval(line.strip())

        # getting features and target
        input_data = []
        target = []
        for i in range(1,len(line)-1):
            input_data.append(line[:i])
            target.append(line[i])

        # writing to file
        with open(output_filepath+f" {file_counter}",'a') as w:
            for j in range(len(input_data)):
                data_pair = f"{input_data[j]} {target[j]}\n"
                w.write(data_pair)
        #updating file counter
        if line_counter%1000000==0:
            file_counter += 1

