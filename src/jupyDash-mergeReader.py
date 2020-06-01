import pickle
import collections as coll

def nestFunc():
    return coll.defaultdict(list)

with open('dash-likelihood', 'rb') as handle:
    likelihood1 = pickle.loads(handle.read())            # likelihood[className][word]   {counted}
with open('dash-priors', 'rb') as handle:
    priors1 = pickle.loads(handle.read())                # prior[className]              {counted}
with open('dash-content', 'rb') as handle:
    content1 = pickle.loads(handle.read())               # content[className][sampleName][words(list)]

# dataset used in jupyter
with open('jupy-likelihood', 'rb') as handle:
    likelihood2 = pickle.loads(handle.read())
with open('jupy-priors', 'rb') as handle:
    priors2 = pickle.loads(handle.read())
with open('jupy-content', 'rb') as handle:
    content2 = pickle.loads(handle.read())

likelihood = dict(list(likelihood1.items()) + list(likelihood2.items()))
priors = dict(list(priors1.items()) + list(priors2.items()))
content = dict(list(content1.items()) + list(content2.items()))

with open('dashjupy-likelihood', 'wb') as handle:
    pickle.dump(likelihood, handle)
with open('dashjupy-priors', 'wb') as handle:
    pickle.dump(priors, handle)
with open('dashjupy-content', 'wb') as handle:
    pickle.dump(content, handle)

print("Succesfully merged files.")