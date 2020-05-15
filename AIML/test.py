import aiml
import os
#register.txt: list of customer ids
#id.text stores order number,item_id, quantity
def showOrders(c_id):
    sess_file=open(str(message)+'.txt')
    sess_file.close()
def addOrders(c_id,):
    sess_file=open(str(message)+'.txt')
    sess_file.close()
def cancelOrders(c_id,order_id):
    sess_file=open(str(message)+'.txt')
    sess_file.close()

os.chdir("../AIML/data/")
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
    register = open("register.txt","w")
    register.write(str(id)+"\n") 
    sess_file=open(str(id)+'.txt','a')
    sess_file.close()
else:
    message=input("Enter Customer id: ")
    id=int(message)
    
register.close()
while True:
    message=input("Enter your message >> ")
    output=kernel.respond(message)

    print(output)
sess_file.close()