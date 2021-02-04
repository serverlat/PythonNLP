# Exercise 1
words = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']

# a
print("Words that start with 'sh':")
for word in words:
    if word[:2] == "sh":
        print(word, end=" ")
print("\n")

# b
print("Words longer than 4 characters:")
for word in words:
    if len(word) > 4:
        print(word, end=" ")
print("\n")
