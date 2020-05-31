#import ipynb.fs.defs.nbsvm_tutorial as nb
import collections as coll
import string, pickle

def nestFunc():
    return coll.defaultdict(list)

stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

def readFile(fileName, trainAmount, testAmount, stopW):
    name = ''
    if fileName == 'yelp_labelled.txt':
        name = 'revs-yelp '
    elif fileName == 'imdb_labelled.txt':
        name = 'revs-imdb '
    elif fileName == 'amazon_cells_labelled.txt':
        name = 'revs-amazon '
    generalize = str.maketrans('', '', string.punctuation) # create removing punctuation parameter for translate method
    priors = coll.Counter()                                # counting dictionary, where non-existing key is 0
    likelihood = coll.defaultdict(coll.Counter)            # counting dictionary(words) within another dictionary(classes)
    content = coll.defaultdict(nestFunc)                   # creating list of words within dictionary(classes)

    with open(fileName, 'r') as f:
        words = []
        for ith, line in enumerate(f):    # line - sample
            if ith == trainAmount:
                break
            line = line.translate(generalize).lower()
            words.append(line.split())
            priors[name + (words[ith][-1])] += 1
            for word in words[ith]:
                if word != '0' and word != '1' and not word in stopW:
                    likelihood[name + words[ith][-1]][word] += 1

        for ith, line in enumerate(f):    # line - sample
            if ith == testAmount:
                break
            line = line.translate(generalize).lower()
            words.append(line.split())
            for word in words[ith]:
                if word != '0' and word != '1' and not word in stopW:
                    content[name + words[ith][-1]][ith].append(word)
            
    return likelihood, priors, content

'''
def main():
    likelihood1, priors1, content1 = readFile('amazon_cells_labelled.txt', 750, 250, stopWords)
    likelihood2, priors2, content2 = readFile('imdb_labelled.txt', 750, 250, stopWords)
    likelihood3, priors3, content3 = readFile('yelp_labelled.txt', 750, 250, stopWords)
    likelihood = dict(list(likelihood1.items()) + list(likelihood2.items()) + list(likelihood3.items()))
    priors = dict(list(priors1.items()) + list(priors2.items()) + list(priors3.items()))
    content = dict(list(content1.items()) + list(content2.items()) + list(content3.items()))
    print("Data reading complete")

    if True:    # WRITING DATA INTO TEXT FILES FOR EXAMINATION
        print(likelihood, file = open("jupy-likelihood-examine.txt", "w"))       # likelihood[className][word]   {counted}
        print(priors, file = open("jupy-priors-examine.txt", "w"))               # prior[className]              {counted}
        print(content, file = open("jupy-content-examine.txt", "w"))             # content[className][sampleName][words(list)]
        print("Data ready for examination")

    if True:
        with open('jupy-likelihood', 'wb') as handle:
            pickle.dump(likelihood, handle)
        with open('jupy-priors', 'wb') as handle:
            pickle.dump(priors, handle)
        with open('jupy-content', 'wb') as handle:
            pickle.dump(content, handle)
        print("Data ready to process!")

main()
'''