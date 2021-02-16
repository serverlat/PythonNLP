import nltk
from nltk.corpus import brown 

#a Most frequent tag

tagged_brown = brown.tagged_words()
tagged_brown_fdist = nltk.FreqDist(tag for (token,tag) in tagged_brown)
most_common = tagged_brown_fdist.most_common(1)
print(f"The most common tag is {most_common[0][0]}, it appears {most_common[0][1]} times.")

#b Ambiguous words

cfdist = nltk.ConditionalFreqDist(tagged_brown)
ambi_words = set([word.lower() for word in cfdist.conditions() if len(cfdist[word]) >= 2]) 
ambiwords_count = len(ambi_words)
print(ambiwords_count, "words are ambigious.")

#c Percentage of ambiguous words

wordcount = len(set([word.lower() for word in brown.words()]))
print(f"There are {wordcount} total words in the Brown corpus, {round(ambiwords_count/wordcount*100, 2)} % of these are ambigious.")

#d Sentences with most ambiguous words

tag_counts = [(word, (len(set(cfdist[word].keys()).union(set(cfdist[word.upper()].keys()))))) for word in ambi_words]
sorted_tag_counts = sorted(tag_counts, key=lambda word: word[1], reverse=True)
top_ten = [tag[0] for tag in sorted_tag_counts][:10]
print(top_ten)
topten_sentences = {}
for tag in top_ten:
    for sentence in brown.sents():
           topten_sentences.setdefault(tag, []).append(" ".join(sentence))
           if len(topten_sentences[tag]) > 4:
               break


