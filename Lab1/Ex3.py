# Exercise 3
import nltk
from nltk.corpus import cmudict as d

dictionary = d.dict()  # the CMU pronouncing dictionary


# syllables function borrowed from:
# https://stackoverflow.com/questions/5876040/number-of-syllables-for-words-in-a-text
def syllables(word):
    if word in dictionary:
        return max([len([syl for syl in entry if syl[-1].isdigit()]) for entry in dictionary[
            word.lower()]])
    else:
        return -1
        # the number of stressed phonems in a word is approximately the number of syllables
        # makes lists of syllables, takes max if there are more than one pronunciation
        # the number of syllables is approximately the number of


# a doesn't work for vowel sounds like honest
def word_to_pig_latin(word):
    piglatinized = ""
    vowels = [index for index, char in enumerate(word) if char.lower() in "aeiou"]
    if word[0].lower() in "aeiou":
        if syllables(word) > 1:
            piglatinized = word[vowels[1]:] + word[:vowels[1]] + "ay"
        else:
            piglatinized = word + "yay"
    else:
        if word[1] in "aeiou":
            piglatinized = word[1:] + word[0] + "ay"
        else:
            if len(word) > 2:
                piglatinized = word[vowels[0]:] + word[:vowels[0]] + "ay"
            else:
                piglatinized = word[-1] + word[:0] + "ay"
    return piglatinized


# b
# not accounting for apostrophes, the word "am" doesnt work
def text_to_pig_latin(text):
    pigtext = ""
    char_indices = [index for index in range(len(text)) if text[index].isalpha() is False]
    tokens = nltk.word_tokenize(text)
    counter = 0
    for word in tokens:
        if word.isalpha():
            newword = word_to_pig_latin(word.lower())
            if word[0].isupper():
                newword = newword[0].upper() + newword[1:]
            pigtext += newword
            counter += len(word)
            while counter in char_indices:
                pigtext += text[counter]
                counter += 1
    return pigtext


# c
# First we have to remove all the "ay" and "yay" from the words. For words than begin with vowels, we're done
# after having removed "yay" (unless the alternative encoding is used, where the first vowel + consonant cluster is
# removed). Then we have to distinguish between words beginning with one consonant or a consonant cluster. The only
# way I can think of is to try both cases (i.e. moving the last letter to the first place and moving the two last
# letters to the first place) and checking if any of the words make sense, which would be very time consuming, at least
# if we're using Wikipedia's definitions of Pig Latin.
# Ex. keyboard -> erboardkay and training -> ainingtray would become either keyboard/dkeyboar and rainingt/training
# and the computer wouldn't know any better which one is correct. We'd also have to account for words that start with
# three consonants, which would add even more ambiguity to the task. I don't know how to do this without human
# intervention or some kind of thesaurus that's available for the computer to check.