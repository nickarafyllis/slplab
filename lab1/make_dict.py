# Loading the package
import spacy
nlp = spacy.load("en_core_web_sm")

#nlp = en_core_web_sm.load()

# Reading a file
myfile = open("corpus.txt").read()

corpus = nlp(myfile)

dict = {}

# Iterate over each token in the document
for token in doc:
    # Check if the token is a punctuation or a whitespace
    if not token.is_space:
        # Increment the frequency of the token in the dictionary
        if token.text not in token_counts:
            dict[token.text] = 1
        else:
            dict[token.text] += 1

# Print the token frequencies
print(dict)

# Create a new dictionary with only the tokens with a count greater than 5
#frequent_tokens = {token: count for token, count in dict.items() if count > 5}

# Print the frequent tokens
#print(frequent_tokens)

# Write the frequent tokens to a file
#with open('vocab/words.vocab.txt', 'w') as f:
#    for token, count in frequent_tokens.items():
#        f.write(f"{token}\t{count}\n")
