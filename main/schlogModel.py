import numpy as np

class schloglModel:
    
    # Define base paramters, based on ODE model
    def__init__(self):
        self.concA = 10.0
        self.concB = 20.0
        self.k1 = 6.0
        self.k2 = 1.0
        self.k3 = 230.0
        self.k4 = 1000.0
        self.vol = 8.0
        self.nreactions = 4
        
    # Define CME birth/death rates
    def lambdan(self,x):
        return concA*k1*x*(x-1)/vol + concB*k3*vol
    def mun(self,x):
        return k2*x*(x-1)*(x-2)/vol**2 + x*k4

    # Define intensity functions for tau-leaping
    def lambda1(self, x):
        return concA*k1*x*(x-1)/vol
    def lambda2(self, x):
        return k2*x*(x-1)*(x-2)/vol**2
    def lambda3(self, x):
        return concB*k3*vol
    def lambda4(self,x):
        return  x*k4    
    
    # Define ODE (LMA) function to explore parameters
    def ODE_func(self, x,k1,k2,k3,k4,a,b):
        return k1*a*x**2 - k2*x**3- k4*x + k3*b
    
    # Calculate nonequilibrium steady state solution
    def steadystate_solution(n):
    result = 1.0
    for i in range(n):
        result = result*(lambdan(i)/mun(i+1))
    return result