from collections import Counter
import spacy
import textacy
from spacy import displacy
import pandas as pd
import numpy as np
from wordfreq import word_frequency
from wordfreq import zipf_frequency
import matplotlib.pyplot as plt
import random
import os
from scipy.stats import pearsonr
import sklearn
import json

print("#######################  LOADING DATA & MODELS")

# Loading data and Model
nlp = spacy.load('en_core_web_sm')
cwd = os.getcwd()
train_data = open(f'{cwd}/data/preprocessed/train/sentences.txt', encoding='utf-8')
text = train_data.read()
train_data_doc = nlp(text)

# TASK 1

print("#######################  TASK 1")

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

#print(word_frequencies)
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


print("#######################  TASK 2")
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


pos_frequencies = Counter(pos_list)
rtf = []
rtf.append(round(pos_frequencies['NOUN, NN']/sum(pos_frequencies.values()), 2))
rtf.append(round(pos_frequencies['PROPN, NNP']/sum(pos_frequencies.values()), 2))
rtf.append(round(pos_frequencies['ADP, IN']/sum(pos_frequencies.values()), 2))
rtf.append(round(pos_frequencies['DET, DT']/sum(pos_frequencies.values()), 2))
rtf.append(round(pos_frequencies['ADJ, JJ']/sum(pos_frequencies.values()), 2))
rtf.append(round(pos_frequencies['NOUN, NNS']/sum(pos_frequencies.values()), 2))
rtf.append(round(pos_frequencies['PUNCT, ,']/sum(pos_frequencies.values()), 2))
rtf.append(round(pos_frequencies['PUNCT, .']/sum(pos_frequencies.values()), 2))
rtf.append(round(pos_frequencies['SPACE, _SP']/sum(pos_frequencies.values()), 2))
rtf.append(round(pos_frequencies['VERB, VBN']/sum(pos_frequencies.values()), 2))

finegrained = ["NOUN", "PROPN", "ADP", "DET", "ADJ", "NOUN", "PUNCT", "PUNCT", "SPACE", "VERB"]
universal = ["NN", "NNP", "IN", "DT", "JJ", "NNS", ",", ".", "_SP", "VBN"]
occurrences = [len(NN_Noun), len(NNP_Propn), len(IN_Adp), len(DT_Det), len(JJ_Adj), len(NNS_Noun), len(COMMA_Punct), len(PERIOD_Punct), len(SP_Space), len(VBN_Verb)]

most_frequent = []
most_frequent.append(Counter(NN_Noun).most_common(3))
most_frequent.append(Counter(NNP_Propn).most_common(3))
most_frequent.append(Counter(IN_Adp).most_common(3))
most_frequent.append(Counter(DT_Det).most_common(3))
most_frequent.append(Counter(JJ_Adj).most_common(3))
most_frequent.append(Counter(NNS_Noun).most_common(3))
most_frequent.append(Counter(COMMA_Punct).most_common(3))
most_frequent.append(Counter(PERIOD_Punct).most_common(3))
most_frequent.append(Counter(SP_Space).most_common(3))
most_frequent.append(Counter(VBN_Verb).most_common(3))

least_frequent = []
least_frequent.append(Counter(NN_Noun).most_common()[-1])
least_frequent.append(Counter(NNP_Propn).most_common()[-1])
least_frequent.append(Counter(IN_Adp).most_common()[-1])
least_frequent.append(Counter(DT_Det).most_common()[-1])
least_frequent.append(Counter(JJ_Adj).most_common()[-1])
least_frequent.append(Counter(NNS_Noun).most_common()[-1])
least_frequent.append(Counter(COMMA_Punct).most_common()[-1])
least_frequent.append(Counter(PERIOD_Punct).most_common()[-1])
least_frequent.append(Counter(SP_Space).most_common()[-1])
least_frequent.append(Counter(VBN_Verb).most_common()[-1])

word_class_table = pd.DataFrame({"Finegrained POS-tag":finegrained, "Universal POS-tag":universal, "Occurrences": occurrences, "Relative Tag Frequency (%)" : rtf, "3 most frequent tokens" : most_frequent, "Example of infrequent token": least_frequent})
word_class_table.head(10)

# This was done in a notebook so the print function in terminal displays it improperly:

print(word_class_table)

print("#######################  TASK 3 with textacy:")

ngrams = list(textacy.extract.basics.ngrams(train_data_doc, 2, filter_stops=False, filter_punct=False))
ngrams = [str(i) for i in ngrams]
ngrams_frequency = Counter(ngrams)
print(ngrams_frequency.most_common(3))


pos_list = [str(token.pos_) for token in train_data_doc] #nlp() needs string
pos_list_string = nlp(' '.join(pos_list)) #textacy.extract.basics.ngrams() needs nlp()-element
ngrams = list(textacy.extract.basics.ngrams(pos_list_string, 2))    #ATTENTION: textacy.extract.basics.ngrams() produces elements with type of "spacy.tokens.span.Span", not "str" -> Counter doesn't work for "spacy.tokens.span.Span"
ngrams = [str(i) for i in ngrams]
ngrams_frequency = Counter(ngrams)
print(ngrams_frequency.most_common(3))


print("#######################  TASK 3 manually without textacy:")

token_bi = []
pos_bi = []
token_tri = []
pos_tri = []


loop_counter = 0



for token in train_data_doc:
  
    try:
        token_bi.append(f"{train_data_doc[loop_counter]}, {train_data_doc[loop_counter + 1]}")
        pos_bi.append(f"{train_data_doc[loop_counter].pos_}, {train_data_doc[loop_counter + 1].pos_}")
    except:
        pass
    try:
        token_tri.append(f"{train_data_doc[loop_counter]}, {train_data_doc[loop_counter + 1]}, {train_data_doc[loop_counter + 2]}")
        pos_tri.append(f"{train_data_doc[loop_counter].pos_}, {train_data_doc[loop_counter + 1].pos_}, {train_data_doc[loop_counter + 2].pos_}")
    except:
        pass
    loop_counter += 1
    
print(f"3 most frequent:\n\nToken Bigrams: {Counter(token_bi).most_common(3)}\nToken Trigrams: {Counter(token_tri).most_common(3)}\nPOS Bigrams: {Counter(pos_bi).most_common(3)}\nPOS Trigrams: {Counter(pos_tri).most_common(3)}")

print("#######################  TASK 3 Generating ngram plot:")

ngrams_bi = dict(Counter(token_bi))
ngrams_tri = dict(Counter(token_tri))
ngrams_sorted_bi = sorted(ngrams_bi.items(), key=lambda x: x[1], reverse=True)
ngrams_sorted_tri = sorted(ngrams_tri.items(), key=lambda x: x[1], reverse=True)

frequencies_bi = []
frequencies_tri = []
for i in range(len(ngrams_sorted_bi)):
    frequencies_bi.append(ngrams_sorted_bi[i][1])
for i in range(len(ngrams_sorted_tri)):
    frequencies_tri.append(ngrams_sorted_tri[i][1])

x = range(11251)
y = range(14264)
plt.plot(frequencies_bi[0:21])
plt.plot(frequencies_tri[0:21])
plt.legend(["Bi-grams", "Tri-grams"])
plt.suptitle('n-gram frequencies')
plt.ylabel("Frequency")
plt.xlabel("Ngrams (20 most frequent)")
plt.xticks(range(0, 21))
plt.show()

print("#######################  TASK 4:")

from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV
import random

random_sentence = random.randint(0, len(list(train_data_doc.sents)) -1)


# DONT RUN WE FOUND A NICE ONE:

# # This gets us a random sentence
# i = 0
# for sentences in train_data_doc.sents:
#     if i == random_sentence:
#         sentence = sentences
#     i += 1

# for token in sentence:
#     if 2 < len(getAllInflections(token.text)) < 4:
#         inflections = getAllInflections(token.text)
#         inflection_list = []
#         for i in range(len(all_inflections)):
#             inflection_list.append(str(all_inflections[i][0]))
#         inflection_set = set(inflection_list)
#         if len(inflection_set) == 3:
#             word_token = token
#             break
#         else:
#             continue

for sentences in train_data_doc.sents:
    sentence = sentences
    for token in sentence:
        if token.text == "say":
            word_token = token
            inflections = getAllInflections(token.text)
            break


print(f"Word: {word_token.text}\nLemma: {word_token.lemma_}\nInflected forms: {inflections}")


all_inflections =  list(inflections.values())

doc_length = 0
for i in train_data_doc.sents:
    doc_length += 1


def check_sentences(inflection):
    for sentences in train_data_doc.sents:
        sentence = str(sentences)
        if str(inflection) in sentence:
            print(f"({inflection}): {sentence}")
            break

inflection_list = []
for i in range(len(all_inflections)):
    inflection_list.append(str(all_inflections[i][0]))
    
inflection_set = set(inflection_list)
for inflection in inflection_set:
    check_sentences(inflection)


print("#######################  TASK 5:")

print(len(train_data_doc.ents))
print(len(Counter([ent.label_ for ent in train_data_doc.ents]).keys()))
print(Counter([ent.label_ for ent in train_data_doc.ents]).keys())
stop_counter = 0
for sentence in train_data_doc.sents:
   stop_counter += 1
   displacy.render(sentence, jupyter=True, style='ent')
   if stop_counter == 5: break


print("#######################  TASK 7:")

#wiki_news_train = open('C:/Users/mpete/OneDrive/Desktop/Uni/Master_DBI/Period_5/NLP/Assignments/Assignment_1/intro2nlp_assignment1_code/data/original/english/WikiNews_Train.tsv', encoding='utf-8')
wiki_news_train_df = pd.read_table(f'{cwd}/data/original/english/WikiNews_Train.tsv', header = None)
wiki_news_train_df.columns = ['ID', 'sentence', 'start_index', 'end_index', 'target_word', 'nat', 'non-nat', 'nat diff', 'non-nat diff', 'binary', 'prob']

target_word = list(wiki_news_train_df["target_word"])
binary_label = list(wiki_news_train_df["binary"])
prob_label = list(wiki_news_train_df["prob"])

token_list = []

for i in range(len(target_word)):
    instance = nlp(target_word[i])
    token_list.append(0)
    for token in instance:
        token_list[i] +=1


print(f"Number of instances labeled with 0: {Counter(binary_label)[0]}\n\
Number of instances labeled with 1: {Counter(binary_label)[1]}\n\
Probabilistic label:\n\
\tmin: {np.min(prob_label)}\n\
\tmax: {np.max(prob_label)}\n\
\tmedian: {np.median(prob_label)}\n\
\tmean: {round(np.mean(prob_label), 3)}\n\
\tstd: {round(np.std(prob_label), 3)}\n\
Number of instances consisting of more than one token: {sum(i > 1 for i in token_list)}\n\
Maximum number of tokens for an instance: : {max(token_list)}")


print("#######################  TASK 8")

length_tokens = []
frequency = []
prob = []
pos = []


#filter only instances with one token and which are labeled as complex from at least one annotator
for index, row in wiki_news_train_df.iterrows():
    instance = nlp(row['target_word'])
    tokens_per_instance = [token for token in instance]
    if (len(tokens_per_instance) == 1) and (row['binary'] == 1):
        length_tokens.append(len(tokens_per_instance[0].text))   
        frequency.append(zipf_frequency(tokens_per_instance[0].text, 'en'))
        prob.append(row['prob'])
        pos.append(tokens_per_instance[0].pos_)

# print("Pearson correlation length and complexity: {}".format(np.corrcoef(length_tokens, prob)))
# print("Pearson correlation frequency and complexity: {}".format(np.corrcoef(frequency, prob)))

print("Pearson correlation length and complexity: {}".format(pearsonr(length_tokens, prob)))
print("Pearson correlation frequency and complexity: {}".format(pearsonr(frequency, prob)))

plt.scatter(length_tokens, prob)
plt.xlabel("Length Tokens")
plt.ylabel("Probabilistic complexity")
plt.show()

plt.scatter(frequency, prob)
plt.xlabel("Frequency")
plt.ylabel("Probabilistic complexity")
plt.show()

plt.scatter(pos, prob)
plt.xlabel("Part-of-Speech")
plt.ylabel("Probabilistic complexity")
plt.show()