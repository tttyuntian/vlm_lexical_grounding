import os
import pandas as pd
import nltk
import re

DATA_ROOT = "../../data/wikiHow/"
DATA_PATH = os.join(DATA_ROOT, "wikihowAll.csv")
TAG_RE = re.compile(r'<[^>]+>')

concat_flag = False  # whether to concatenate two sentences together
output_path = os.path.join(DATA_ROOT, "wikihowAll_clean_single.csv")
txt_output_path = os.path.join(DATA_ROOT, "failure_report_single.txt")



def remove_tags(text):
    return TAG_RE.sub('', text)

def preprocess_text(sentence):
    # Removing html tags
    sentence = remove_tags(sentence)
    
    # Remove punctuations and digits, except ".!?[]"
    #sentence = re.sub('[^a-zA-Z\.\?\!]', ' ', sentence)
    
    # Remove single character
    #sentence = re.sub(r'\s+[a-Az-Z]\s+', ' ', sentence)
    
    # Replace \n to space
    sentence = sentence.replace("\n", " ")

    # Remove spaces
    sentence = sentence.strip()

    return sentence

if __name__ == '__main__':
    # Read data
    data = pd.read_csv(DATA_PATH)

    # Process the dataset to 1-sentence samples with [CLS] and [SEP]
    res = []
    sent_tokenize_fails = 0
    process_fails = 0

    for i in range(data.shape[0]):
        try:
            text = data.iloc[i]['text']
            sentences = nltk.sent_tokenize(text)
        except:
            sent_tokenize_fails += 1
            continue
        
        # Every sentence is a sample
        for sentence in sentences:
            try:
                sentence = preprocess_text(sentence)
                if len(sentence) == 0:
                    continue
            except:
                process_fails +=1
                continue
            res.append("[CLS] {} [SEP]".format(sentence))

    # Save as .csv file
    res = pd.DataFrame({'text': res})
    res.to_csv(output_path, index = False)
    
    # Report failures
    with open(txt_output_path, 'w') as f:
        f.write('nltk.sent_tokenize() failures: {}\n'.format(sent_tokenize_fails))
        f.write('preprocess_text() failures: {}'.format(process_fails))
        f.close()
