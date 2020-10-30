import csv
import numpy as np
import itertools
import nltk
# nltk.download('punkt')

def GetSentenceData(path, vocabDim=8000):
    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    tokenizedSentences = [nltk.word_tokenize(sent) for sent in sentences]
    # Filter the sentences having few words (including SENTENCE_START and SENTENCE_END)
    tokenizedSentences = list(filter(lambda x: len(x) > 3, tokenizedSentences))

    # Count the word frequencies
    wordFreq = nltk.FreqDist(itertools.chain(*tokenizedSentences))
    print("Found %d unique words tokens." % len(wordFreq.items()))

    # Get the most common words and build index2word and word2index vectors
    vocab = wordFreq.most_common(vocabDim-1)
    index2word = [x[0] for x in vocab]
    index2word.append(unknown_token)
    word2index = dict([(w,i) for i,w in enumerate(index2word)])

    print("Using vocabulary size %d." % vocabDim)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenizedSentences):
        tokenizedSentences[i] = [w if w in word2index else unknown_token for w in sent]

    print("\nExample sentence: '%s'" % sentences[1])
    print("\nExample sentence after Pre-processing: '%s'\n" % tokenizedSentences[0])

    # Create the training data
    XTrain = np.asarray([[word2index[w] for w in sent[:-1]] for sent in tokenizedSentences])
    yTrain = np.asarray([[word2index[w] for w in sent[1:]] for sent in tokenizedSentences])

    print("XTrain shape: " + str(XTrain.shape))
    print("yTrain shape: " + str(yTrain.shape))

    # Print a training data example
    xExample, yExample = XTrain[17], yTrain[17]
    print("x:\n%s\n%s" % (" ".join([index2word[x] for x in xExample]), xExample))
    print("\ny:\n%s\n%s" % (" ".join([index2word[x] for x in yExample]), yExample))

    return XTrain, yTrain


  