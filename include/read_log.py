f=open('log.out')
g=open('1.txt','w')
CHAR=[str(i) for i in range(10)]+["#"]
l=f.readline()
while l:
    if l[0] not in CHAR:
        g.write(l)
    l=f.readline()
f.close()
g.close()
