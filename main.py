from os import listdir
from os.path import join

from NaiveBayes.Pool import Pool

documentClasses = ["politik",  "sport",  "wirtschaft"]
base = "data/"
pool = Pool()
for documentClass in documentClasses:
    learnDirectory = join(base, documentClass, "train")
    pool.learn(learnDirectory, documentClass)
documentTags = {}
for documentClass in documentClasses:
    testDirectory = join(base, documentClass, "test")
    dir = listdir(testDirectory)
    for file in dir:
        probabilities = pool.Probability(join(testDirectory, file))
        probability = sorted(probabilities, reverse=True, key=lambda p: p[1])[0]
        documentTags[file] = str(probability)


for file, tag in sorted(documentTags.items()):
    print(file + " : " + str(tag))


