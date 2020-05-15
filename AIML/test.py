import aiml
import os
#register.txt: list of customer ids
#id.text stores order number,item_id, quantity
def showOrders(c_id):
    sess_file=open(str(c_id)+'.txt')
    print("OrderID-------------Item_Name")
    print(sess_file.read())
    sess_file.close()
def addOrders(c_id):
    order_id=0
    sess_file=open(str(c_id)+'.txt','r')
    for _ in sess_file:
        order_id += 1
    sess_file.close()
    sess_file=open(str(c_id)+'.txt','a')
    item_name=str(input("Enter Order: "))
    sess_file.write(str(order_id)+","+str(item_name)+"\n")
    sess_file.close()
    print("Order with id: "+str(order_id)+" has been placed")
def cancelOrders(c_id):
    item_name=int(input("Enter Order ID to cancel: "))
    sess_file=open(str(c_id)+'.txt','r')
    orders=sess_file.readlines()
    sess_file.close()
    sess_file=open(str(c_id)+'.txt','w')
    for idnum in range(len(orders)):
        if idnum!=order_id:
            sess_file.write(orders[idnum])
    print("Order with id: "+str(order_id)+" has been cancelled")

os.chdir("../AIML/data/")
register=open('register.txt','r')
kernel=aiml.Kernel()
kernel.learn("std-startup.xml")
kernel.respond("load aiml b")

message=input("Hello this is Alice, a sales bot, Are you a new customer?(y/n): ")
if message=='y':
    c_id=0
    for x in register:
        c_id+=1
    register.close()
    register = open("register.txt","w")
    register.write(str(id)+"\n") 
    sess_file=open(str(id)+'.txt','a')
    sess_file.close()
else:
    message=input("Enter Customer id: ")
    c_id=int(message)
register.close()

while True:
    message=input("Enter your message >> ")
    output=kernel.respond(message)
    if output=="show":
        print("Showing Orders")
        showOrders(c_id)
    if output=="place":
        print("Placing Order")
        addOrders(c_id)
    if output=="cancel":
        print("Cancelling Orders")
        cancelOrders(c_id)
    else:
        print(output)
sess_file.close()