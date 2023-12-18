x = [0,1,0,3,12]

for i in range(len(x)):
    if x[i] == 0:
        x.append(x.pop(i))

print(x)