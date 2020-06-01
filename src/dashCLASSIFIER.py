import collections as coll
import math, time, pickle

# dash doesnt like lambda operators
def nestFunc():
    return coll.defaultdict(list)
def nestFunc2():
    return coll.defaultdict(dict)

# classify one sample
def classifySample(contentSample, probabs, priors, zeroFix):
    maxPC = -1E6, ''                # maxP = - 100 000 (for init), maxC = category with max probability
    probProcess = {}                # probability process dictionary for each class

    if zeroFix != 1:                # rational number fix
        for category in probabs:
            p = math.log(priors[category])                              # sum of all samples in the current category
            n = float(sum(probabs[category].values()))                  # sum of all words in the current category (trained, multiple ones too)

            probProcess[category] = []
            probProcess[category].append(p)                             # first value is prior
            for j, word in enumerate(contentSample):
                probProcess[category].append(math.log(max(zeroFix, probabs[category][word] / n)))   # log because of zero converging
                p += probProcess[category][j+1]

            if p > maxPC[0]:
                maxPC = p, category
            probProcess[category].append(p)                             # saving final probability for ternary visualization of samples 
    else:                           # laplace smoothing
        for category in probabs:
            p = math.log(priors[category])
            n = float(sum(probabs[category].values()))
            
            probProcess[category] = []
            probProcess[category].append(p)
            for j, word in enumerate(contentSample):
                probProcess[category].append(math.log((probabs[category][word]+1) / n))
                p += probProcess[category][j]
            if p > maxPC[0]:
                maxPC = p, category
            probProcess[category].append(p)
    return maxPC[1], probProcess

def classifyTestSet(likelihood, priors, content, zeroFix):
    tSamples = 0
    tCorrect = 0
    skewInfo = coll.defaultdict(coll.Counter)               # skewing info - skewInfo[class]['TP'/'FP'/'FN'/'TN'/'support']
    probProcess = coll.defaultdict(nestFunc2)               # calculation process of predictions for visualizations

    for className in content:
        for sample in content[className]:
            skewInfo[className]['support'] += 1
            classPrediction, probProcess[className][sample] = classifySample(content[className][sample], likelihood, priors, zeroFix)
            probProcess[className][sample]
            if classPrediction == className:
                tCorrect += 1
                skewInfo[className]['TP'] += 1
                for classRep in content:
                    if classRep != className:
                        skewInfo[classRep]['TN'] += 1
            else:
                skewInfo[classPrediction]['FP'] += 1
                for classRep in content:
                    if classRep == className:
                        skewInfo[classRep]['FN'] += 1
                    elif classRep != classPrediction:             
                        skewInfo[classRep]['TN'] += 1
            tSamples += 1

    accuracy = [tCorrect, tSamples]
    return accuracy, skewInfo, probProcess

# calculating precision, recall, average, F1-score, support (number of samples)
def calcReport(skewInfo):
    tableResult = {}
    summaryReport = [0, 0, 0, 0, 0]             # adding all calculations for avg/total
    #print("\t"*5 + "%-13s%-12s%-12s%-12s%-12s" % ("Precision", "Recall", "Average", "F1-score", "Support"))
    i = 0
    for className in skewInfo:
        if skewInfo[className]['TP'] != 0:
            precision = skewInfo[className]['TP'] / (skewInfo[className]['TP'] + skewInfo[className]['FP'])
            recall = skewInfo[className]['TP'] / (skewInfo[className]['TP'] + skewInfo[className]['FN'])
            f1 = (2*precision*recall)/(precision+recall)
        else:
            precision = 0
            recall = 0
            f1 = 0
        avg = (precision + recall) / 2
        summaryReport[0] += precision
        summaryReport[1] += recall
        summaryReport[2] += avg
        summaryReport[3] += f1
        summaryReport[4] += skewInfo[className]['support']
        i += 1
        #print("%-43s%-12.2f%-12.2f%-12.2f%-12.2f%-12i" % (className, round(precision,2), round(recall,2), round(avg,2), round(f1,2), skewInfo[className]['support']))
        tableResult[className] = [round(precision,2), round(recall,2), round(avg,2), round(f1,2), skewInfo[className]['support']]

    #print("\n%-43s%-12.2f%-12.2f%-12.2f%-12.2f%-12i" % ("Avg/total", round(summaryReport[0]/i,2), round(summaryReport[1]/i,2), round(summaryReport[2]/i,2), round(summaryReport[3]/i,2), summaryReport[4]))
    tableResult["Avg/total"] = [round(summaryReport[0]/i,2), round(summaryReport[1]/i,2), round(summaryReport[2]/i,2), round(summaryReport[3]/i,2), summaryReport[4]]
    
    return tableResult

'''
def main():
    classifyAmount = 1
    accSum = 0
    
    for i in range(0, classifyAmount):

        with open('dash-likelihood', 'rb') as handle:
            likelihood = pickle.loads(handle.read())            # likelihood[className][word]   {counted}
        with open('dash-priors', 'rb') as handle:
            priors = pickle.loads(handle.read())                # prior[className]              {counted}
        with open('dash-content', 'rb') as handle:
            content = pickle.loads(handle.read())               # content[className][sampleName][words(list)]

        accuracy, skewInfo = classifyTestSet(likelihood, priors, content)
        accSum += accuracy
        calcReport(accuracy, skewInfo)

    accSum /= (i+1)
    print((accSum, classifyAmount, (end - start)), file = open("dash-NB.txt", "a"))             # content[className][sampleName][words(list)]

main()
'''