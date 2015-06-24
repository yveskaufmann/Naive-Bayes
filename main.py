from os import listdir
from os.path import join

from NaiveBayes.Pool import Pool

documentClasses = ["politik",  "sport",  "wirtschaft"]
base = "data/"
pool = Pool()
for documentClass in documentClasses:
    learnDirectory = join(base, documentClass, "train")
    pool.learn(learnDirectory, documentClass)

for documentClass in documentClasses:
    testDirectory = join(base, documentClass, "test")
    dir = listdir(testDirectory)
    for file in dir:
        res = pool.Probability(join(testDirectory, file))
        print(documentClass + ": " + file + ": " + str(res))

