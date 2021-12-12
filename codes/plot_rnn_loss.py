import matplotlib.pyplot as plt

with open("./loss.txt",'r')as f:
    data=f.read()

data=data[1:-1]
data=data.split(',')
data=[eval(x) for x in data]
print(data)
plt.plot(data)
plt.show()