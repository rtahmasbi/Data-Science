
from sklearn.datasets import load_boston
import gzip

boston = load_boston()

# save in .gz format
f = gzip.GzipFile('boston.gz', "w")
np.save(f, boston.data)
f.close()

# save in .txt format
np.savetxt('boston2.txt',boston.data)


### to read data
f = gzip.GzipFile('boston.gz', "r")
data = np.load(f)
f.close()


data = np.loadtxt('boston2.txt')





#
