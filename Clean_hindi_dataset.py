#!/usr/bin/env python
# coding: utf-8

# # Clean Hindi Dataset

# In[219]:


import string
import re
import time
import os


# In[220]:


# Retrieves one paragraph or sentence as given in the dataset
def reader(fp):
    rs = ' '
    sent = ''

    while rs:
        rs = fp.readline()
        if rs == '\n':
            y = sent
            sent = ''
            yield y
        else:
            sent += rs.replace('\n',' ')
    y = sent
    sent = ''
    yield sent


# In[221]:


# Cleans the hindi string passed
def cleaner(s):
    
    #removing symbols or numbers or english letters from the beginning of a sentence
    s = re.sub(r"^[!\"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~/a-zA-Z\s0-9\*]*",'',s)
    
    #removing symbols or numbers or english words given inside brackets () or {} or []
    s = re.sub(r"(\(|\[|\{)[!\"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~/\w\s\d\*]*(\}|\)|\])",'',s)
    
    #removing symbols or numbers or english letters from the ending of a sentence
    s = re.sub(r"[-\w\d_:/\\][!\"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~/a-zA-Z\s0-9\*]*$",'',s)
    
    #removing all characters other than devanagiri and english, like unicode characters and symbols other than punctuations
    #("\u0900-\u097f" represents all devanagiri script)
    s = re.sub(r"[^\w\s\d\u0900-\u097F!\"$%&\'()*+,-./:;=?@[\\\]_`{|}]+",'',s)
    
    #Remove space from left side of .
    s = re.sub(r"\s+\.",'.',s)
    
    #change . to | for sentence ending
    #remove multiple dots (...)
    s = re.sub(r"\.\.+",r".",s)
    #first, for . which have no space after the . we will add a space
    s = re.sub(r"([\u0900-\u097F])\.([\u0900-\u097F])",r"\1। \2",s)
    #second, for other . (we have to exclude decimal numbers)
    s = re.sub(r"([^\d])\.([^\d])",r"\1। \2",s)
    
    #stripping spaces at beginning and end of sentence
    s = s.strip()
    
    #add | if the sentence does not have | at the end of it.
    s = re.sub(r"((?![।])[\u0900-\u097F\d])$",r"\1।",s)
    
    #remove extra spaces between words
    s = ' '.join(s.split())
    
    
    return s


# In[222]:


if __name__ == "__main__":
    
    folderpath = "/media/dai/New Volume/Dataset/"
    input_filename = "hindi.txt"
    output_filename = "clean-hindi"
    input_filepath = folderpath+input_filename
    output_filepath = folderpath+output_filename
    
    # English letters a-z,A-Z
    letters = list(string.ascii_letters)
    
    #Swear words
    swords = "आंड़,आंड,आँड,बहनचोद,बेहेनचोद,भेनचोद,बकचोद,बकचोदी,बेवड़ा,बेवड़े,बेवकूफ,भड़ुआ,भड़वा,भोसड़ा,भोसड़ीके,भोसड़ीकी,भोसड़ीवाला,भोसड़ीवाले,बब्बे,बूबे,बुर,चरसी,चूचे,चूची,चुची,चोद,चुदने,चुदवा,चुदवाने,चाट,चूत,चूतिया,चुटिया,चूतिये,दलाल,दलले,फट्टू,गधा,गधे,गधालंड,गांड,गांडू,गंडफट,गंडिया,गंडिये,गू,गोटे,हग,हग्गू,हगने,हरामी,हरामजादा,हरामज़ादा,हरामजादे,हरामज़ादे,हरामखोर,झाट,झाटू,कुत्ता,कुत्ते,कुतिया,कुत्ती,लेंडी,लोड़े,लौड़े,लौड़ा,लोड़ा,लौडा,लिंग,लोडा,लोडे,लंड,लौंडा,लौंडे,लौंडी,लौंडिया,लुल्ली,मार,मारो,मारूंगा,मादरचोद,मादरचूत,मादरचुत,मम्मे,मूत,मुत,मूतने,मुतने,मूठ,मुठ,नुननी,नुननु,पाजी,पेसाब,पेशाब,पिल्ला,पिल्ले,पिसाब,पोरकिस्तान,रांड,रंडी,सुअर,सूअर,टट्टे,टट्टी,उल्"
    swear_words = set(swords.split(','))
    
    #To time the run
    start_time = time.time()
    with open(output_filepath,"w") as wp:
        with open(input_filepath,"r") as fp:
            excluded_lines = 0
            line_counter = 0

            for s in reader(fp):
                s = cleaner(s)
                line_counter += 1

                #tokenizing sentence to check for swear words
                ss = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~।]+',' ',s)

                #checking if there are swear words or english words in the sentence, if not write to new file
                if (set(ss.split()).intersection(swear_words) == set()) and (not any([True for x in letters if x in s])):
                    wp.write(s+"\n\n")
                else:
                    excluded_lines += 1

                #Just printing * to get approximate time for execution and approximate idea of how much processing completed
                if line_counter%1000000 == 0:
                    print("*",end=' ')
                    if line_counter%10000000 == 0:
                        print()

    # Printing the total lines in the dataset and the lines excluded and the time of the run
    print(f"\nTotal lines: {line_counter}")
    print(f"Total lines with english words: {excluded_lines}")
    end_time = time.time()
    print(f"Time taken: {end_time-start_time:.2f} seconds")

    
    

