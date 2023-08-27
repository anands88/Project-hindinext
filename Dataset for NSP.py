from Clean_hindi_dataset import reader

folderpath = "/media/dai/New Volume/Dataset/"
input_filename = "clean-hindi-dataset"
output_filename = "NSP-dataset"
input_filepath = folderpath+input_filename
output_filepath = folderpath+output_filename



line_counter = 0
excluded_lines = 0

with open(output_filepath,'w') as wp:
    with open(input_filepath,'r') as fp:
        
        for s in reader(fp):
            line_counter += 1
            
            #splitting paragraphs into sentences
            sentences = s.split("।")
            
            #sentences when split would have an empty string at the end so we use len > 2
            if len(sentences)>2:
                wp.write(s+"\n\n")
            else:
                excluded_lines += 1

            #For visualizing the running process
            if line_counter%1000000 == 0:
                    print("*",end=' ')
                    if line_counter%10000000 == 0:
                        print()
                        
        print(f"\nTotal lines: {line_counter}")
        print(f"Total lines with english words: {excluded_lines}")
            
