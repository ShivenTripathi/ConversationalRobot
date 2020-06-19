req=open("../requirements.txt")
cleaned=[]
for line in req:
    flag=0
    #print(line[100])
    for i in range(len(line)-1):
        if(line[i]=='=' and line[i+1]== '='):
            flag+=1
        if flag==2:
            line=line[:i]
            break
    cleaned.append(line)
req.close()
# print(cleaned)
f = open("req.txt", "a")
for line in cleaned:
    f.write(line+"\n")
f.close()

