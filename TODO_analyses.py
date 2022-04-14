from collections import Counter
import spacy
nlp = spacy.load('en_core_web_sm')

train_data = open('C:/Users/mpete/OneDrive/Desktop/Uni/Master_DBI/Period_5/NLP/Assignments/Assignment_1/intro2nlp_assignment1_code/data/preprocessed/train/sentences.txt', encoding='utf-8')
text = train_data.read()
train_data_doc = nlp(text)
'''
#----------------------------------------------------------------------------------------------------------------------
#TASK 1

#tokens = [token.text for token in train_data_doc]
#print(tokens)
#print(len(train_data_doc))

word_frequencies = Counter()
words_per_sentence = []

for sentence in train_data_doc.sents:
    words = []
    words_single_sentence = 0
    for token in sentence:
        # Let's filter out punctuation and new lines ("/n")
        if (not token.is_punct) and (not token.is_space):
            words.append(token.text)
            words_single_sentence += 1
    words_per_sentence.append(words_single_sentence)
    word_frequencies.update(words)

print(word_frequencies)
num_tokens = len(train_data_doc)
num_words = sum(word_frequencies.values())
num_types = len(word_frequencies.keys())
avg_num_words_per_sentence = sum(words_per_sentence) / len(words_per_sentence)
avg_word_length = sum(len(word) for word in words) / len(words)

print('Number of Tokens: ' + str(num_tokens))
print('Number of Words: ' + str(num_words))
print('Number of Types: ' + str(num_types))

print('Average number of words per sentence: {number:.{digits}f}'.format(number = avg_num_words_per_sentence, digits = 2))
print('Average word length: {number:.{digits}f}'.format(number = avg_word_length, digits = 2))

#----------------------------------------------------------------------------------------------------------------------
'''
#TASK 2

pos_list = []

NN_Noun = []
NNP_Propn = []
IN_Adp = []
DT_Det = []
JJ_Adj = []
NNS_Noun = []
COMMA_Punct = []
PERIOD_Punct = []
SP_Space = []
VBN_Verb = []

for token in train_data_doc:
    #print(token.pos_, token.tag_)
    pos_list.append('{}, {}'.format(token.pos_, token.tag_))
    if ('{}, {}'.format(token.pos_, token.tag_) == 'NOUN, NN'):
        NN_Noun.append(token.text)

    elif ('{}, {}'.format(token.pos_, token.tag_) == 'PROPN, NNP'):
        NNP_Propn.append(token.text)

    elif ('{}, {}'.format(token.pos_, token.tag_) == 'ADP, IN'):
        IN_Adp.append(token.text)

    elif ('{}, {}'.format(token.pos_, token.tag_) == 'DET, DT'):
        DT_Det.append(token.text)

    elif ('{}, {}'.format(token.pos_, token.tag_) == 'ADJ, JJ'):
        JJ_Adj.append(token.text)

    elif ('{}, {}'.format(token.pos_, token.tag_) == 'NOUN, NNS'):
        NNS_Noun.append(token.text)

    elif ('{}, {}'.format(token.pos_, token.tag_) == 'PUNCT, ,'):
        COMMA_Punct.append(token.text)

    elif ('{}, {}'.format(token.pos_, token.tag_) == 'PUNCT, .'):
        PERIOD_Punct.append(token.text)

    elif ('{}, {}'.format(token.pos_, token.tag_) == 'SPACE, _SP'):
        SP_Space.append(token.text)

    elif ('{}, {}'.format(token.pos_, token.tag_) == 'VERB, VBN'):
        VBN_Verb.append(token.text)

'''
pos_frequencies = Counter(pos_list)
print(pos_frequencies)
print(round(pos_frequencies['NOUN, NN']/sum(pos_frequencies.values()), 2))
print(round(pos_frequencies['PROPN, NNP']/sum(pos_frequencies.values()), 2))
print(round(pos_frequencies['ADP, IN']/sum(pos_frequencies.values()), 2))
print(round(pos_frequencies['DET, DT']/sum(pos_frequencies.values()), 2))
print(round(pos_frequencies['ADJ, JJ']/sum(pos_frequencies.values()), 2))
print(round(pos_frequencies['NOUN, NNS']/sum(pos_frequencies.values()), 2))
print(round(pos_frequencies['PUNCT, ,']/sum(pos_frequencies.values()), 2))
print(round(pos_frequencies['PUNCT, .']/sum(pos_frequencies.values()), 2))
print(round(pos_frequencies['SPACE, _SP']/sum(pos_frequencies.values()), 2))
print(round(pos_frequencies['VERB, VBN']/sum(pos_frequencies.values()), 2))
'''

print(Counter(NN_Noun))
print(Counter(NNP_Propn))
print(Counter(IN_Adp))
print(Counter(DT_Det))
print(Counter(JJ_Adj))
print(Counter(NNS_Noun))
print(Counter(COMMA_Punct))
print(Counter(PERIOD_Punct))
print(Counter(SP_Space))
print(Counter(VBN_Verb))

