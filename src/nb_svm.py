import collections as coll
import string, os, math, random, time, pickle

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import metrics

# calculate number amounts of samples for each category
def cSetAmountSamples(dataFolder, applyMinMax, pickPercentage):

    # finding minMax amount of samples from all classes
    minMaxS = 10000                                     # presuming that any of given classes has no less than 10000 samples
    totalS = []
    i = 0
    for className in os.listdir(dataFolder):
        classPath = dataFolder + "/" + className        # getting into class folder
        total = 0
        for sample in os.listdir(classPath):            # counting samples
            total += 1
        if minMaxS > total:
            minMaxS = total
        totalS.append(total)                            # saving amount of samples for each class into the list 
        i += 1
    
    if applyMinMax:                                 
        totalS = []
        part = (minMaxS * pickPercentage) // 100
        for j in range(0, i):
            totalS.append(part)
    else:
        for j in range(0, i):
            totalS[j] = (totalS[j] * pickPercentage) // 100

    print("minMax samples in category:", minMaxS)
    print("Sample amount in each class:", totalS)

    return totalS

def nestFunc():
    return coll.defaultdict(list)

# read data from given dictionary parameter argument
def readData(par):                                                  # par - dictionary with parameters of sample picking
    generalize = str.maketrans('', '', string.punctuation)          # create removing punctuation parameter for translate method

    priors = coll.Counter()                                         # counting dictionary, where non-existing key is 0
    likelihood = coll.defaultdict(coll.Counter)                     # counting dictionary(words) within another dictionary(classes)
    content = coll.defaultdict(nestFunc)      # creating list of words within dictionary(classes)

    i = 0
    for className in os.listdir(par['dataFolder']):                 # reading each class folder
        print("Reading " + className)
        classPath = par['dataFolder'] + "/" + className             # getting inside's folder location
        trainAmount = par['ratio'] * par['totalS'][i] // 100        # calculate the ratio of train/test samples 

        j = 0
        cSamples = os.listdir(classPath)
        if par['rand']:
            random.shuffle(cSamples)                                # shuffling data for random picking
        for sample in cSamples:                     # reading samples
            if j == par['totalS'][i]:                               # read until the parameter total sample amount is satisfied
                break
            samplePath = par['dataFolder'] + "/" + os.path.join(className, sample)  # getting inside's text folder location
            f = open(samplePath, 'r')                               # opening text sample folder

            if trainAmount > j:                     # train
                priors[className] += 1                              # adding new sample occurence to get total amount of samples for each class
                for word in f.read().split():                       # f.read = reads all data, .split splits them into single words
                    word = word.translate(generalize).lower()       # remove punctuation and lower all letters
                    if not word in par['stopWords'] and word != '': # ignore empty strings and stopWords
                        likelihood[className][word] += 1            # add occurence into word dictionary within class dictionary
            else:                                   # test
                for word in f.read().split():
                    word = word.translate(generalize).lower()
                    if not word in par['stopWords'] and word != '':
                        content[className][sample].append(word)     # adding new word into sample list of words 

            f.close()
            j += 1
        i += 1

    return likelihood, priors, content

#maxR = 0
#maxPA = 2500
def classifyTestSet(likelihood, priors, content):
    tSamples = 0
    tCorrect = 0
    classInfoPR = coll.defaultdict(coll.Counter)            # classInfoPR[class]['TP'/'FP'/'FN'/'TN'] (skewing info)
    for className in content:
        print("Classifying samples in " + className)
        for sample in content[className]:
            classInfoPR[className]['support'] += 1
            prediction = classifySample(content[className][sample], likelihood, priors)
            if prediction == className:
                tCorrect += 1
                classInfoPR[className]['TP'] += 1
                for classRep in content:
                    if classRep != className:
                        classInfoPR[classRep]['TN'] += 1
            else:
                classInfoPR[prediction]['FP'] += 1
                for classRep in content:
                    if classRep == className:
                        classInfoPR[classRep]['FN'] += 1
                    elif classRep != prediction:             
                        classInfoPR[classRep]['TN'] += 1
            tSamples += 1

    #global maxR
    #global maxPA
    #print("max rozdiel: {} maxPA: {}".format(maxR, maxPA))
    accuracy = tCorrect/tSamples
    return accuracy, classInfoPR                            

''' #       P(    A|B) = P(B|A)     * P(A)     / P(B)               # Bayes rule
    #       posterior  = likelihood * priorC    / priorP            # terminology

    # posterior                 - P(C|obs)                  # P conditioned on observations (need to find max)
    # likelihood                - P(obs|C)  *               # word frequency in category
    # prior                     - P(C)      /               # category frequency (no P, but just counted cats)
    # prior (total words obs)   - P(obs)      - 4.          # all possible words (const, redundant)
    # P(obs)          - all observations (in each category its the same, no need it then)
    # obs             - observations (words in sample)

    #   P(C|obs)    = P(obs|C)                               * P(C) / P(obs)
    #               = argmax(C) -||-(obs) P(w|Cw)            * P(C) 
    #               = argmax(C) -||-(obs) P(w !U Cw) / P(Cw) * P(C)
    #                                     word in sample intersected with number of occurences in category  /
    #                                     number of all words in category (including multiples)             *
    #                                     number of samples in category
    #               = argmax(C) -||-(obs)    (probabs[category][word] / n) * p
''' #               = argmax(C) -||-(obs) log(probabs[category][word] / n) + log(p)

# classify one sample
def classifySample(contentSample, probabs, priors):
    #global maxR
    #global maxPA
    max_p = -1E7, ''
    #min_p = 0.0000
    for category in probabs:
        p = math.log(priors[category])                              # sum of all samples in the current category
        #minP = math.log(priors[category])
        n = float(sum(probabs[category].values()))                  # sum of all words in the current category (trained, multiple ones too)
        for word in contentSample:
            #print("prior: {}, likelihood: {}".format(p, math.log(max(1E-8, probabs[category][word] / n))))
            #minP += math.log(1E-8)
            p += math.log(max(1E-8, probabs[category][word] / n))   # max(1E-8, 0) - for removing zeros, log for converging to zero via many multiplications
        if p > max_p[0]:
            max_p = p, category
        #if p < min_p:
        #    min_p = p
        #rozdiel = max_p[0] - min_p
        #if rozdiel > maxR or maxR is None:
        #    maxR = rozdiel
            #print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS {}".format(maxR))
        #if maxR > 500:
        #if maxPA > p - minP:
        #    maxPA = p - minP
        #print("p: {0:.10f}        minP: {1:.10f}      rozdiel: {2:.10f}     p-minP: {3:.10f}".format(p, minP, rozdiel, p-minP))

    #print("Max: {}".format(max_p[0]))
    return max_p[1]

# calculating precision, recall, average, F1-score
def printClassifyReport(accuracy, classInfoPR):
    print("Accuracy:", accuracy)
    summaryReport = [0, 0, 0, 0, 0]
    print("                            %-13s%-12s%-12s%-12s%-12s" % ("Precision", "Recall","Average", "F1-score", "Support"))
    i = 0
    for className in classInfoPR:
        if classInfoPR[className]['TP'] != 0:
            precision = classInfoPR[className]['TP'] / (classInfoPR[className]['TP'] + classInfoPR[className]['FP'])
            recall = classInfoPR[className]['TP'] / (classInfoPR[className]['TP'] + classInfoPR[className]['FN'])
            f1 = (2*precision*recall)/(precision+recall)
        else:
            precision = 0
            recall = 0
            f1 = 0
            print("WTF just happened")
        avg = (precision + recall) / 2
        #f1 = (2*precision*recall)/(precision+recall)
        summaryReport[0] += precision
        summaryReport[1] += recall
        summaryReport[2] += avg
        summaryReport[3] += f1
        summaryReport[4] += classInfoPR[className]['support']
        i += 1
        print("%-30s%-12.2f%-12.2f%-12.2f%-12.2f%-12i" % (className, round(precision,2), round(recall,2), round(avg,2), round(f1,2), classInfoPR[className]['support']))

    print("\n%-30s%-12.2f%-12.2f%-12.2f%-12.2f%-12i" % ("Avg/total", round(summaryReport[0]/i,2), round(summaryReport[1]/i,2), round(summaryReport[2]/i,2), round(summaryReport[3]/i,2), summaryReport[4]))
    print()

def main():
    # PARAMETERS
    dataFolder = "ohsumed-all"
    ratio = 75                                              # pick 75% of samples into training set and 20% into test set
    randomness = True                                       # shuffling train/test data
    SVMon = False
    
    applyMinMax = False                                     # find minimal max amount of samples from all classes (making it as max amount of picking data for all classes)
    numOfClasses = 3                                       # picking data from 3 first classes in dataFolder directory
    pickPercentage = 1                                      # how much % of data from each category we want to pick
    classifyAmount = 1                                     # classification amount times for averaging accuracy

    totalS = cSetAmountSamples(dataFolder, applyMinMax, pickPercentage)   # Calculate totalS - count list of samples for each category

    parameters = {  'dataFolder' : dataFolder,
                    'totalS' : totalS,
                    'ratio' : ratio,
                    'rand' : randomness,
                    'stopWords' : []
                    }
    '''
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                                    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'
    '''

    start = time.time()

    # READING DATA
    accSum = 0
    for i in range(0, classifyAmount):
        if True:
            likelihood, priors, content = readData(parameters)
            # WRITING DATA INTO TEXT FILES FOR EXAMINATION
            #print(likelihood, file = open("likelihood.txt", "w"))       # likelihood[className][word]   {counted}
            #print(priors, file = open("priors.txt", "w"))               # prior[className]              {counted}
            #print(content, file = open("content.txt", "w"))             # content[className][sampleName][words(list)]

        if True:
            print("start dumping")
            with open('likelihood.txt', 'wb') as handle:
                pickle.dump(likelihood, handle)
            with open('priors.txt', 'wb') as handle:
                pickle.dump(priors, handle)
            with open('content.txt', 'wb') as handle:
                pickle.dump(content, handle)

        if False:
            with open('likelihood.txt', 'rb') as handle:
                likelihood = pickle.loads(handle.read())
            with open('priors.txt', 'rb') as handle:
                priors = pickle.loads(handle.read())
            with open('content.txt', 'rb') as handle:
                content = pickle.loads(handle.read())

        print("end dumping")

        accuracy, classInfoPR = classifyTestSet(likelihood, priors, content)
        accSum += accuracy
        printClassifyReport(accuracy, classInfoPR)


    accSum /= (i+1)
    end = time.time()

    if SVMon:
        print("SVM Classifier")
        vect = CountVectorizer(stop_words = parameters['stopWords'], encoding='latin-1')    # if problem with decoding -> decode_error='ignore'
        dataset = load_files(dataFolder, shuffle=False)     # shuffle makes target_names correspond to target numbers(which are defaultly shuffled)

        clf = LinearSVC(C=1.00)         # the larger C, the more fitness into training data -> overfitting
        pickPercentage /= 100                                    
        ratio = (pickPercentage * ratio) / 100
        accSum2 = 0
        start2 = time.time()
        for i in range(0, classifyAmount):
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(vect.fit_transform(dataset.data), dataset.target, 
                                        test_size = (pickPercentage - ratio), train_size = ratio, random_state = None)
    
        #print(vect.get_feature_names())
        #print(Xtrain.shape, Ytrain.shape)
        #print(Xtest.shape, Ytest.shape)
        #print(Xtrain.toarray())                # bag of words, row -> sample, column -> feature (occurences)
        #print(Ytrain)                          # vector of Y to particular rows of X
                                                # vect.get_feature_names() - words which were converted into X via occurence numbers
            clf.fit(Xtrain,Ytrain)

            predic = clf.predict(Xtest)
            accuracy = accuracy_score(Ytest, predic)
            accSum2 += accuracy
            print(accuracy)
            print(metrics.classification_report(Ytest, predic, target_names = dataset.target_names))

        end2 = time.time()
        accSum2 /= (i+1)
        print((pickPercentage, accSum, classifyAmount, dataFolder, (end - start)), file = open("NB.txt", "a"))             # content[className][sampleName][words(list)]
        print((pickPercentage, accSum2, classifyAmount, dataFolder, (end2 - start2)), file = open("SVM.txt", "a"))
    

main()


#ohsumed-all - [2540, 1171, 427, 6327, 1678, 2989, 526, 2589, 715, 3851, 998, 2518, 1623, 6102, 1277, 1086, 1617, 1919, 865, 3116, 2933, 506, 9611]
#ohsumed-all-4smaller - [2540, 1171, 427, 1678, 526]
#20news-18828 - [799, 973, 985, 982, 961, 980, 972, 990, 994, 994, 999, 991, 981, 990, 987, 997, 910, 940, 775, 628]
#20news-18828-6smaller - [973, 972, 994, 990, 997, 910]
