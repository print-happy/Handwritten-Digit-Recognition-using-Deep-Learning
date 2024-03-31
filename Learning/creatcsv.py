import os
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'programmers.csv')
with open(data_file, 'w') as f:
    f.write('Programmers,salary,marriage,age\n') # 列名
    f.write('A,10000,NaN,35\n') # 每行表示一个数据样本
    f.write('B,12000,married,37\n')
    f.write('C,NaN,NaN,30\n')
    f.write('D,16000,married,40\n')
    f.write('E,18000,married,45\n')
    f.write('F,20000,NaN,41\n')
    f.write('G,22000,married,50\n')
    f.write('H,24000,NaN,55\n')
    f.write('I,26000,married,60\n')
