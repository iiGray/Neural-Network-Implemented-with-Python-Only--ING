#!/usr/bin/env python
# coding: utf-8
#author: iiGray


import numpy as np,random,time
import matplotlib.pyplot as plt



def f(x):
    return x*np.sin(x)+x


lst=[i/100 for i in range(-300,300)]

l=[f(i) for i in lst]


plt.plot(lst,l)
plt.show()


n=100
m=100
dots=[]
random.seed(time.time()*time.time())
while n or m:
    r=(random.random()*9-3)
    if n==0 and r<0:continue
    if m==0 and r>0:continue
    if r<=-3 or r>4:continue
#     if abs(f(r))<10:continue
    y=random.random()*10-5
    yy=y+f(r)
    if( abs(y)<0.8):continue
    if r<0:n-=1
    else:m-=1
    

    a=0 if yy<f(r) else 1
    
    dots+=[[r,yy,a]]
    


dots=[[(round(dot[0],2),round(dot[1],2)),dot[-1]] for dot in dots]

dots0=[[dot[0][0],dot[0][1]] for dot in dots if dot[-1]==0]
dots0x=[k[0] for k in dots0]
dots0y=[k[1] for k in dots0]

dots1=[[dot[0][0],dot[0][1]] for dot in dots if dot[-1]==1]
dots1x=[k[0] for k in dots1]
dots1y=[k[1] for k in dots1]


plt.scatter(dots0x,dots0y,c="b")
plt.scatter(dots1x,dots1y,c="g")

plt.show()
