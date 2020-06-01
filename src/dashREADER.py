import collections as coll
import string, os, random, pickle

def nestFunc():
    return coll.defaultdict(list)

# calculate number of samples for each category
def csSetAmount(dataFolder, applyMinMax, pickPercentage):

    # finding minMax amount of samples from all classes
    minMaxS = 10000                                     # choosing 10000 number of samples if there is class with the least amount of samples with more than 10000 samples
    csAmounts = []                                      # class sample amounts
    i = 0
    for className in os.listdir(dataFolder):
        classPath = dataFolder + "/" + className
        sAmount = len(os.listdir(classPath))            # sample amount for particular class
        if minMaxS > sAmount:
            minMaxS = sAmount
        csAmounts.append(sAmount)                       # saving amount of samples for each class into the list 
        i += 1
    
    if applyMinMax:
        part = (minMaxS * pickPercentage) // 100        # calculating partial amount of samples
        csAmounts = []
        [csAmounts.append(part) for j in range(0, i)]   # creating list of classes with the same number of samples 
    else:
        for j in range(0, i):
            csAmounts[j] = (csAmounts[j] * pickPercentage) // 100   # calculating sample amounts for each class into the list

    print("minMax samples in category: {} (100% pick percentage)".format(minMaxS))
    print("Sample amount in each class:", csAmounts)
    return csAmounts

# read data from given parameters
def readData(pars):                                                 # par - dictionary with parameters of sample picking
    generalize = str.maketrans('', '', string.punctuation)          # create removing punctuation parameter for translate method

    priors = coll.Counter()                                         # counting dictionary, where non-existing key is 0
    likelihood = coll.defaultdict(coll.Counter)                     # counting dictionary(words) within another dictionary(classes)
    content = coll.defaultdict(nestFunc)                            # creating list of words within dictionary(classes)

    i = 0
    for className in os.listdir(pars['dataFolder']):                # reading each class folder
        print("Reading " + className)
        classPath = pars['dataFolder'] + "/" + className
        trainAmount = pars['ratio'] * pars['csAmounts'][i] // 100   # calculate the ratio of train/test samples 

        j = 0
        cSamples = os.listdir(classPath)
        if pars['rand']:
            random.shuffle(cSamples)                                # shuffling data for random picking
        for sample in cSamples:                                     # reading samples
            if j == pars['csAmounts'][i]:                           # read until the parameter class sample amount is satisfied
                break
            samplePath = pars['dataFolder'] + "/" + os.path.join(className, sample)
            f = open(samplePath, 'r')                               # opening text sample folder

            if trainAmount > j:                         # train
                priors[className] += 1                              # adding new sample occurence to get total amount of samples for each class
                for word in f.read().split():                       # f.read = reads all data, .split splits them into single words
                    word = word.translate(generalize).lower()       # remove punctuation and lower all letters
                    if not word in pars['stopWords'] and word != '':# ignore empty strings and stopWords
                        likelihood[className][word] += 1            # add occurence into word dictionary within class dictionary
            else:                                       # test
                for word in f.read().split():
                    word = word.translate(generalize).lower()
                    if not word in pars['stopWords'] and word != '':
                        content[className][sample].append(word)     # adding new word into sample list of words 
            f.close()
            j += 1

        i += 1

    return likelihood, priors, content

def main():

    # PARAMETERS
    dataFolder = "dash-dataset"
    pickPercentage = 25                                     # pick 25% of samples for each category from the whole dataset
    ratio = 75                                              # pick 75% of samples into training set and 25% into test set
    randomness = False                                      # shuffling train/test data
    applyMinMax = False                                     # find minimal amount of samples and set it as general picking amount for all classes
    stopWords = False                                       # ignore certain words
    parameters = {  
        'dataFolder' : dataFolder,
        'csAmounts' : csSetAmount(dataFolder, applyMinMax, pickPercentage),
        'ratio' : ratio,
        'rand' : randomness
    }
    if stopWords:
        parameters['stopWords'] = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    else:
        parameters['stopWords'] = []

    # READING DATA
    likelihood, priors, content = readData(parameters)
    print("Data reading complete")

    if True:    # WRITING DATA INTO TEXT FILES FOR EXAMINATION
        print(likelihood, file = open("dash-likelihood-examine.txt", "w"))       # likelihood[className][word]   {counted}
        print(priors, file = open("dash-priors-examine.txt", "w"))               # prior[className]              {counted}
        print(content, file = open("dash-content-examine.txt", "w"))             # content[className][sampleName][words(list)]
        print("Data ready for examination")

    if True:
        with open('dash-likelihood', 'wb') as handle:
            pickle.dump(likelihood, handle)
        with open('dash-priors', 'wb') as handle:
            pickle.dump(priors, handle)
        with open('dash-content', 'wb') as handle:
            pickle.dump(content, handle)
        print("Data ready to process!")

main()
