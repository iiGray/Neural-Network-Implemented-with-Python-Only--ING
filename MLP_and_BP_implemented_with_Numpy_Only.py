#!/usr/bin/env python
# coding: utf-8
#author: iiGray

import numpy as np
import math,pickle,time
import matplotlib.pyplot as plt
from collections import defaultdict
from abc import ABC,abstractmethod,abstractproperty


def add_grad(func):
    def inner(self,*args,**kwargs):
        ret=func(self,*args,**kwargs)
        ret.detach=False
        ret.grad=np.zeros(ret.shape)
        return ret
    return inner

def add_grad_inplace(func):
    def inner(self, *args, **kwargs):
        grad = self.grad
        ret = Tensor(func(self, *args, **kwargs))
        ret.grad = getattr(grad, func.__name__)(*args, **kwargs)
        return ret
    return inner


class Tensor(np.ndarray):
    def __new__(cls, input_array, requires_grad=True):
        if type(input_array) == tuple:
            obj = np.random.randn(*input_array).view(cls)
        else:
            obj = np.asarray(input_array).view(cls)
        obj.grad = np.zeros(obj.shape)
        return obj

    @add_grad
    def mean(self, *args, **kwargs):
        return super().mean(*args, **kwargs)

    @add_grad
    def std(self, *args, **kwargs):
        return super().std(*args, **kwargs)

    @add_grad
    def sum(self, *args, **kwargs):
        return super().sum(*args, **kwargs)

    @add_grad
    def __add__(self, *args, **kwargs):
        return super().__add__(*args, **kwargs)

    @add_grad
    def __radd__(self, *args, **kwargs):
        return super().__radd__(*args, **kwargs)

    @add_grad
    def __sub__(self, *args, **kwargs):
        return super().__sub__(*args, **kwargs)

    @add_grad
    def __rsub__(self, *args, **kwargs):
        return super().__rsub__(*args, **kwargs)

    @add_grad
    def __mul__(self, *args, **kwargs):
        return super().__mul__(*args, **kwargs)

    @add_grad
    def __rmul__(self, *args, **kwargs):
        return super().__rmul__(*args, **kwargs)

    @add_grad
    def __pow__(self, *args, **kwargs):
        return super().__pow__(*args, **kwargs)

    @add_grad
    def __rtruediv__(self, *args, **kwargs):
        return super().__rtruediv__(*args, **kwargs)

    @add_grad
    def __truediv__(self, *args, **kwargs):
        return super().__truediv__(*args, **kwargs)

    @add_grad
    def __matmul__(self, *args, **kwargs):
        return super().__matmul__(*args, **kwargs)

    @add_grad
    def __rmatmul__(self, *args, **kwargs):
        return super().__rmatmul__(*args, **kwargs)

    @add_grad_inplace
    def view(self, *args, **kwargs):
        return super().view(*args, **kwargs)

    @add_grad_inplace
    def reshape(self, *args, **kwargs):
        return super().reshape(*args, **kwargs)

    @add_grad_inplace
    def __getitem__(self, *args, **kwargs):
        return super().__getitem__(*args, **kwargs)

    @property
    def zero_grad_(self):
        self.grad = np.zeros(self.grad.shape)

    @property
    def grad_fn(self):
        return "Leaf" if self.detach else "Node"

    def detach_(self, whether=True):
        self.detach = whether


def exp(x):
    return Tensor(np.exp(np.array(x)))


def log(x):
    return Tensor(np.log(np.array(x)))




class MyTorch(ABC):pass
class AbsNet(MyTorch):
    @abstractmethod
    def __init__(self,*args,**kwargs):pass
    @abstractmethod
    def __call__(self,*args,**kwargs):pass
    @abstractmethod
    def forward(self,*args,**kwargs):pass
    @abstractmethod
    def backward(self,*args,**kwargs):pass

class AbsActivation(AbsNet):
    def __init__(self,*args,**kwargs):pass
    @abstractmethod
    def function(self,*args,**kwargs):pass
    def __call__(self,x):
        self.input=x
        self.output=self.forward(x)
        return self.output
    @property
    def zero_grad_(self):
        if "input" in self.__dict__.keys():
            self.input.zero_grad_
            
class AbsOptimizer(MyTorch):
    @abstractmethod
    def __init__(self,*args,**kwargs):pass
    @abstractmethod
    def step(self,*args,**kwargs):pass
    def zero_grad(self):
        self.parameters.zero_grad_       

class AbsModule(AbsNet):
    @abstractproperty
    def zero_grad_(self):pass
    @abstractproperty
    def __repr__(self):pass
    
class AbsLoss(AbsNet):
    @abstractproperty
    def outgrad(self):pass
    def backward(self):
        cgrad=self.outgrad
        for block_name,block in reversed(self.net.__dict__.items()):
            if type(block).__base__ not in (AbsActivation,AbsModule,Module,):
                continue
            cgrad=block.backward(cgrad)



class Linear(AbsModule):
    def __init__(self,in_features,out_features,bias=True):
        self.in_features=in_features
        self.out_features=out_features
        self.bias=bias
        '''使用和torch.nn.Linear中一样参数初始化:参数a=sqrt(5),mode='fan_in'的kaiming_uniform_初始化'''
        bound=1/math.sqrt(in_features)
        
        self.parameters={"weight":Tensor((np.random.rand(in_features,out_features)-0.5)*2*bound)}
        if bias:
            self.parameters["bias"]=Tensor((np.random.rand(1,out_features)-0.5)*2*bound)
            
    def __call__(self,x):
        self.input=x
        self.output=self.forward(x)
        return self.output
    
    def forward(self,x):
        out=x @ self.parameters["weight"]
        if self.bias:
            out+=self.parameters["bias"]
        return out
    
    def backward(self,cgrad):
        try:
            self.input.grad= cgrad @ self.parameters["weight"].T
        except AttributeError:
            raise AttributeError("The layer: "+self.__repr__()+" absent from FP!")
        self.parameters["weight"].grad+= self.input.T @ cgrad
        if self.bias:
            self.parameters["bias"].grad+= cgrad.sum(0,keepdims=True)
        return self.input.grad.copy()
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, "+\
        f"out_features={self.out_features}, bias={self.bias})"
    
    @property
    def zero_grad_(self):
        if "input" in self.__dict__.keys():
            self.input.zero_grad_
        self.parameters["weight"].zero_grad_
        if self.bias:
            self.parameters["bias"].zero_grad_
            

class Sigmoid(AbsActivation):
    def function(self,x):
        return 1/(1+exp(-x))
    
    def forward(self,x):
        return self.function(x)
    
    def backward(self,cgrad):
        assert self.output.shape==cgrad.shape,"Activation Sigmoid BP Error!"
        try:
            self.input.grad=(self.output*(1-self.output))*cgrad
        except (AttributeError):
            raise AttributeError("Layer: " +self.__repr__()+" absent from FP!")
        return self.input.grad
    
    def __repr__(self):
        return "Sigmoid()"
    
        
class Tanh(AbsActivation):
    def function(self,x):
        return (1-exp(-2*x))/(1+exp(-2*x))
    
    def forward(self,x):
        return self.function(x)
    
    def backward(self,cgrad):
        assert self.output.shape==cgrad.shape,"Activation Tanh BP Error!"
        try: 
            self.input.grad=(1-self.output**2)*cgrad
        except (AttributeError):
            raise AttributeError("Layer: " +self.__repr__()+" absent from FP!")
        return self.input.grad
    def __repr__(self):
        return "Tanh()"



class Module(AbsModule):
    def __init__(self,*args,**kwargs):
        raise NotImplementedError("Class: \"Module\" has to be overrided!")
        
    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)
    
    def forward(self,*args,**kwargs):
        raise NotImplementedError("Function: \"forward()\" has to be overloaded!")
        
    def backward(self,cgrad):
        for block_name,block in reversed(self.__dict__.items()):
            if type(block).__base__ not in (AbsActivation,AbsModule,Module):continue
            cgrad=block.backward(cgrad)
        return cgrad
    def __repr__(self):
        name="Net(\n"
        for block_name,block in self.__dict__.items():
            if type(block).__base__ not in (AbsNet,AbsActivation,AbsModule,):
                continue
            name+="  ("+str(block_name)+"): "+block.__repr__()+"\n"
        return name+")"
    @property
    def zero_grad_(self):
        for block_name,block in self.__dict__.items():
            if type(block).__base__ not in (AbsActivation,AbsModule,Module):continue
            block.zero_grad_


class BCEWithLogitsLoss(AbsLoss):
    def __init__(self,net,reduction="none"):
        self.net=net
        self.reduction=reduction
        self.function=Sigmoid()
        
    def __call__(self,y,y_hat):
        return self.forward(y,y_hat)
    
    def forward(self,y,y_hat):
        self.out=y
        self.hat=y_hat
        p=self.function(y)
        ret=-(y_hat*log(p)+(1-y_hat)*log(1-p))
        if self.reduction=="mean":return ret.mean()
        elif self.reduction=="sum":return ret.sum()
        return ret
    
    @property
    def outgrad(self):
        out=self.out
        hat=self.hat
        out.grad=(self.function(out)-hat)/out.shape[0]
        return out.grad
            

class Mini_BGD(AbsOptimizer):
    def __init__(self,net,lr=0.001):
        self.parameters=net
        self.lr=lr
    def step(self):
        for block_name,block in reversed(self.parameters.__dict__.items()):
            if type(block).__base__!=AbsModule:continue
            for name,weight in block.parameters.items():
                weight-=self.lr*weight.grad      



class AdamW(AbsOptimizer):
    def __init__(self,net,lr=0.01,betas=(0.9,0.999),eps=1e-08,weight_decay=0.01):
        self.parameters=net
        self.lr=lr
        self.betas=betas
        self.eps=eps
        self.weight_decay=weight_decay
        

        self.t=0
        self.mt=defaultdict(dict)
        self.vt=defaultdict(dict)
        for block_name,block in reversed(self.parameters.__dict__.items()):
            if type(block).__base__!=AbsModule:continue
            for name,weight in block.parameters.items():
                self.mt[block_name][name]=np.zeros_like(weight)
                self.vt[block_name][name]=np.zeros_like(weight)
    def step(self):
        beta1,beta2=self.betas
        self.t+=1
        for block_name,block in self.parameters.__dict__.items():
            if type(block).__base__!=AbsModule:continue
            for name,weight in block.parameters.items():
                gt=weight.grad
                mt=self.mt[block_name][name]
                vt=self.vt[block_name][name]
                
                weight-=self.lr*gt
                
                self.mt[block_name][name]=beta1*mt+(1-beta1)*gt
                self.vt[block_name][name]=beta2*vt+(1-beta2)*(gt*gt)
                
                mt=mt/(1-np.power(beta1,self.t))
                vt=vt/(1-np.power(beta2,self.t))
                
                weight-=self.lr*mt/(np.sqrt(vt)+self.eps)
                
        
class DataLoader:
    def __init__(self,dataset,batch_size):
        self.dataset=dataset
        self.batch_size=batch_size
        self.num=0
        self.stop=False
        self.final=False
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.final==True:
            self.num=0
            self.final=False
            
        if not self.stop:
            bs=self.batch_size
            num=self.num
            
            self.num=min(self.num+bs,len(self.dataset))
            if self.num==len(self.dataset):self.stop=True
            return [Tensor(np.stack([self.dataset[i][j] 
                                     for i in range(num,self.num)]))
                    for j in range(2)]
        
        self.stop=False
        self.final=True
        raise StopIteration
        


class Dataset:
    def __init__(self,data):
        self.data=data if type(data)==list else pickle.load(open(data,"rb"))
        
    def __getitem__(self,i):
        return \
    np.array(list(self.data[i][0])),np.array([self.data[i][1]],dtype=np.int32)
    
    def __len__(self):
        return len(self.data)


class Net(Module):
    def __init__(self,in_dim):
        self.linear1=Linear(in_dim,5)
        self.tanh1=Tanh()
        
        self.linear2=Linear(5,3)
        self.tanh2=Tanh()
        
        self.linear3=Linear(3,1)
        
    def forward(self,x):
        out=self.linear1(x)
        out=self.tanh1(out)
        
        out=self.linear2(out)
        out=self.tanh2(out)
        
        out=self.linear3(out)
        
        return out


def draw(data_pth):
    dots=pickle.load(open(data_pth,"rb"))
    
    dots0=[[dot[0][0],dot[0][1]] for dot in dots if dot[-1]==0]
    dots0x=[k[0] for k in dots0]
    dots0y=[k[1] for k in dots0]
    
    dots1=[[dot[0][0],dot[0][1]] for dot in dots if dot[-1]==1]
    dots1x=[k[0] for k in dots1]
    dots1y=[k[1] for k in dots1]
    
    plt.scatter(dots0x,dots0y,c="g")
    plt.scatter(dots1x,dots1y,c="b")


def train(net,dataloader,epochs,lr,eps=1e-5):

#     optimizer=Mini_BGD(net,lr=lr)
    optimizer=AdamW(net,lr=lr)

    l=BCEWithLogitsLoss(net,reduction="mean")
    
    loss_lst=[100]

    for _ in range(epochs):

        for x,y in dataloader:

            y_pre=net(x)

            loss=l(y_pre,y)

            loss_lst+=[loss]

            l.backward()

            optimizer.step()

            optimizer.zero_grad()
            

        if abs(loss_lst[-1]-loss_lst[-2])<eps:
            break
            

    print("Update times:",len(loss_lst))
    print("Final_Loss:",loss_lst[-1])
    
    plt.xlabel("update_times")
    plt.ylabel("loss")
    plt.plot(loss_lst)
    plt.show()

    
    
def predict(net,test_dataset,trn_path=False,eps=0.001):

    TP,TN,FP,FN=0,0,0,0
    ALL=len(test_dataset)
    datas=[[[],[]],[[],[]]]
    
    
    for i in range(ALL):
        dot,l=test_dataset[i]
        dot=dot[None,:]
        l=l[None,:]
        out=net(dot)[0][0]
        pre=0 if out <0 else 1
        
        datas[pre][0]+=[dot[0][0]]
        datas[pre][1]+=[dot[0][1]]
        
        if l==1:
            if out<0:FP+=1
            else:TP+=1
        else:
            if out<0:TN+=1
            else: FN+=1
    if trn_path:
        draw(trn_path)
        

    plt.scatter(datas[0][0],datas[0][1],c="y")
    plt.scatter(datas[1][0],datas[1][1],c="r")
    plt.show()

    print("Precision:\t",(TP+eps)/(TP+FP+eps))

    print("Recall:\t",(TP+eps)/(TP+FN+eps))

    print("Accuracy:\t",(TP+TN+eps)/(TP+TN+FP+FN+eps))
        

if __name__=="__main__":

    trn_dataset=Dataset("trn_datas.pkl")
    tst_dataset=Dataset("tst_datas.pkl")


    np.random.seed(0)



    net=Net(in_dim=2)
    dataloader=DataLoader(trn_dataset,batch_size=50)



    print(net)


    train(net,dataloader,epochs=200,lr=0.001)


    predict(net,trn_dataset)


    draw("trn_datas.pkl")


    predict(net,tst_dataset,trn_path="trn_datas.pkl")

