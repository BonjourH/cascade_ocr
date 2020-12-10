f=open('1.txt')
g=open('final.txt','w')
CHAR=[str(i) for i in range(10)]+["#"]+['E']
l=f.readline()
while l:
    if l[0] not in CHAR:
        g.write(l)
    l=f.readline()
f.close()
g.close()
