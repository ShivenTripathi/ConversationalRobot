import aiml
import os
os. chdir("../AIML/data/")
register=open('register.txt','r')
kernel=aiml.Kernel()
kernel.learn("std-startup.xml")
kernel.respond("load aiml b")
message=input("Hello this is Alice, a sales bot, Are you a new customer?(y/n): ")
if message=='y':
    id=0
    for x in register:
        id+=1
    register.close()
    register = open("register.txt","w")#write mode 
    register.write(str(id)+"\n") 
    sess_file=open(str(id)+'.txt','a')
else:
    message=input("Enter Customer id: ")
    sess_file=open(str(message)+'.txt','a')
register.close()
while True:
    message=input("Enter your message >> ")
    output=kernel.respond(message)
    print(output)
sess_file.close()