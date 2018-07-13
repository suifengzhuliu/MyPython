from pandas import read_csv
from matplotlib import pyplot



dataset = read_csv('savior.csv', header=0, index_col=0)
values = dataset.values
pyplot.plot(values[:,0])
pyplot.legend()
pyplot.show()