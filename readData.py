import pandas as pd, numpy as np, os, sys
from scipy import stats

dName = 'Improvement5/'

if len(sys.argv) > 1:
    dName = sys.argv[1] + '/'

path = 'data/' + dName
files = os.listdir(path)
dataSet = []
for f in files:
    fname = path + f
    #print(fname)
    dataSet.append(pd.read_csv(fname))

df = pd.concat(dataSet, keys = np.arange(len(files)).astype('int'))

print('file count: ',len(files))

for col in df.columns:
    print(col, round(df[col].mean(),4), round(df[col].std(),4))

a1win = df['A1 Wins']
a2win = df['A2 Wins']

tScore = abs(a1win.mean() - a2win.mean())/(np.sqrt(a1win.var()+a2win.var())/2 * np.sqrt(1/5))
degree = 18
p = 1 - stats.t.cdf(tScore,df=degree)
print("t = " + str(round(tScore,3)))
print("p = " + str(round(2*p,3)))