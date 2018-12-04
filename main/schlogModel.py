import numpy as np
from joblib import Parallel, delayed
import multiprocessing

class schloglModel:
    
    def __init__(self):
        ''' Define base paramters, based on ODE model and data generation parameters'''
        self.setModelParameters(10., 20., 6., 1., 230., 1000., 8.)
        self.setDataParameters(0, 500, 0.0001, 10, 1000, 2560, 5, 50)

        
    def setModelParameters(self, concA, concB, k1, k2, k3, k4, vol):
        self.concA = concA
        self.concB = concB
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.vol = vol
        self.nreactions = 4


    def setDataParameters(self, xmin, xmax, dt, stride, timesteps, datasize, datamultiplier, outresolution):
        self.xmin = xmin
        self.xmax = xmax
        self.dt = dt
        self.stride = stride
        self.timesteps = timesteps
        self.datasize = datasize
        self.datamultiplier = datamultiplier
        self.outresolution = outresolution
        self.filename =  "data/schlogl_data_vol" + str(self.vol) + "_ndata" + str(self.datasize) + ".dat"
        
        
    def lambdan(self,x):
        '''Define CME birth rate '''
        return self.concA*self.k1*x*(x-1)/self.vol + self.concB*self.k3*self.vol

    def mun(self,x):
        '''Define CME death rate '''
        return self.k2*x*(x-1)*(x-2)/self.vol**2 + x*self.k4


    # Define intensity functions for each reaction (for tau-leaping)
    def lambda1(self, x):
        return self.concA*self.k1*x*(x-1)/self.vol
    def lambda2(self, x):
        return self.k2*x*(x-1)*(x-2)/self.vol**2
    def lambda3(self, x):
        return self.concB*self.k3*self.vol
    def lambda4(self,x):
        return  x*self.k4    


    def ODE_func(self, x,k1,k2,k3,k4,a,b):
        '''Define ODE (LMA) function to explore parameters '''
        return k1*a*x**2 - k2*x**3- k4*x + k3*b


    def steadystate_solution(self, n):
        '''Calculate nonequilibrium steady state solution'''
        result = 1.0
        for i in range(n):
            result = result*(self.lambdan(i)/self.mun(i+1))
        return result
    
    def drift(self, x):
        ''' Define drift of corresponding Chmical Langevin equation (CLE)'''    
        return self.lambda1(x) - self.lambda2(x) + self.lambda3(x) - self.lambda4(x)

    def diffusion(self, x):
        ''' Define diffusion of corresponding Chmical Langevin equation (CLE)'''    
        sigma = [np.sqrt(self.lambda1(x)), - np.sqrt(self.lambda2(x)), 
                np.sqrt(self.lambda3(x)), -np.sqrt(self.lambda4(x))]
        return np.array(sigma)
    
    def sigma(self, x):
        ''' Define coefficient in Chemical Fokker-Planck equation (CFPE); tensor flow version'''
        return self.lambda1(x) + self.lambda2(x) + self.lambda3(x) + self.lambda4(x)
    
    

 # MORE GENERAL CASES   
 
     #def drift_tf(self, x):
        #''' Define drift of corresponding Chmical Langevin equation (CLE)'''    
        #drift = [self.lambda1(x) - self.lambda2(x) + self.lambda3(x) - self.lambda4(x)]
        #return tf.stack(drift)
    
    #def sigma(self, x):
        #''' Define matrix in Chemical Fokker-Planck equation (CFPE)'''
        #diff = self.diffusion(x)
        #return np.tensordot(diff,diff.transpose(), 1) 
    
    #def diffusion_tf(x):
        #''' Define diffusion of corresponding Chemical Langevin equation (CLE); tensor flow version'''
        #sig = [tf.sqrt(lambda1(x)), -1* tf.sqrt(lambda2(x)), tf.sqrt(lambda3(x)), -1* tf.sqrt(lambda4(x))]
        #sigma = tf.stack(sig)
        #return sigma
    
    #def sigma_tf(x):
        #''' Define matrix in Chemical Fokker-Planck equation (CFPE); tensor flow version'''
        #diff = diffusiontf(x)
        #return diff * tf.transpose(diff) 
       



    def terminalCondition(self, n):
        '''Define terminal condition for learning CME of Schlogl model'''
        hist = np.linspace(self.xmin, self.xmax, self.outresolution)
        dx = hist[1] - hist[0]
        #result = []
        for i in range(len(n)):
            index = np.where((n[i] >= hist) & (n[i] < hist + dx ))[0][0]
            iresult = np.zeros(self.outresolution)
            iresult[index] = 1.0
            #if index == 0:
                #iresult[index] = 0.9
                #iresult[index+1] = 0.1
            #elif index == len(n):
                #iresult[index-1] = 0.1
                #iresult[index] = 0.9
            #else:
                #iresult[index-1] = 0.1
                #iresult[index] = 0.8
                #iresult[index+1] = 0.1
            #result.append(iresult)
        return iresult


    def integrateTauLeap(self, X0):
        '''Do tau-leap integration (only for scalar initial condition)'''
        N = np.zeros(self.timesteps)
        N[0] = X0
        for i in range(self.timesteps-1):
            numReactions1 = np.random.poisson(self.lambda1(N[i])*self.dt, 1)
            numReactions2 = np.random.poisson(self.lambda2(N[i])*self.dt, 1)
            numReactions3 = np.random.poisson(self.lambda3(N[i])*self.dt, 1)
            numReactions4 = np.random.poisson(self.lambda4(N[i])*self.dt, 1)
            N[i+1] = N[i] + numReactions1 - numReactions2 + numReactions3 - numReactions4
            if N[i+1] < 0:
                N[i+1] = 0
        return N

    def integrateGillespie(self, X0):
        '''Do Gillespie integration loop (only for scalar initial condition)'''
        N = np.zeros(1)
        N[0] = X0
        t = 0.0
        i = 0
        tfinal = self.timesteps*self.dt
        while t <= tfinal:
            r1 = np.random.rand()
            r2 = np.random.rand()
            rates = [self.lambda1(N[i]), self.lambda2(N[i]), self.lambda3(N[i]), self.lambda4(N[i])]
            lambda0 = np.sum(rates)
            ratescumsum = np.cumsum(rates)
            # Gillespie, time and transition
            lagtime = np.log(1.0/r1)/lambda0
            state = int(sum(r2*lambda0>ratescumsum)) + 1
            if state == 1 or state == 3:
                nextN = N[i] + 1
                N = np.append(N, nextN)
            else: # state == 2 or state == 4:
                nextN = N[i] - 1
                N = np.append(N, nextN)
            t = t + lagtime
            i = i + 1
            print(len(N))
        return N


    def propagate(self, x):
        ''' Propagate using tau-leap or SSA, function definition so the code can be run in parallel'''
        xt = np.float32(self.integrateTauLeap(x))
        y0 = np.zeros(int(self.timesteps/self.stride))
        #xt = np.float32(self.integrateGillespie(x))
        #y0 = np.zeros(int(len(xt)/self.stride))
        for j in range(len(y0)):
            y0[j] = 1.0*xt[j*self.stride]
        return y0


    def generateData(self):
        ''' Generate data and save to file. All particles starting at some x0 
        and propagated for self.timesteps timesteps'''
        x0 = np.float32(np.random.randint(self.xmin, self.xmax, self.datasize))
        num_cores = multiprocessing.cpu_count() 
        results = Parallel(n_jobs=num_cores, verbose = 2)(delayed(self.propagate)(i) for i in x0)
        print("Writing to file ...", end="\r")
        f = open(self.filename, "w")
        for i in range(len(results)):
            f.write(" ".join(str(x) for x in results[i]) + "\n")
        f.close()
        print("Percentage finished:", 100, "%    ", end="\r")
        

    def loadData(self, filename):
        ''' Load data: inputData (x0,tend), targetData u at time tend given x0 at t0=0'''
        targetData = [None]*(self.datasize*self.datamultiplier)
        data = np.genfromtxt(filename, delimiter=' ')
        inputData = data[:,0]
        inputData = np.array([inputData]*self.datamultiplier).flatten()
        tstep = np.random.randint(0, int(self.timesteps/self.stride), self.datasize*self.datamultiplier)
        inputData = np.reshape(np.column_stack((inputData,tstep)), [-1,2])
        for i in range(self.datasize*self.datamultiplier):
            j = i % self.datasize
            targetData[i] = self.terminalCondition(np.array([data[j,tstep[i]]]))
        return inputData, targetData
    
    # Alternative approach for data generation, use initial position and time as input data
    # and propagate until reaching desired fixed time.

    def integrateTauLeap2(self, X0, tsteps):
        '''Do tau-leap integration (only for scalar initial condition)'''
        N = np.zeros(tsteps + 1)
        N[0] = X0
        for i in range(tsteps):
            numReactions1 = np.random.poisson(self.lambda1(N[i])*self.dt, 1)
            numReactions2 = np.random.poisson(self.lambda2(N[i])*self.dt, 1)
            numReactions3 = np.random.poisson(self.lambda3(N[i])*self.dt, 1)
            numReactions4 = np.random.poisson(self.lambda4(N[i])*self.dt, 1)
            N[i+1] = N[i] + numReactions1 - numReactions2 + numReactions3 - numReactions4
            if N[i+1] < 0:
                N[i+1] = 0
        return N


    def propagate2(self, x, tsteps):
        ''' Propagate using tau-leap or SSA, function definition so the code can be run in parallel'''
        xt = np.float32(self.integrateTauLeap2(x, tsteps))
        y0 = np.zeros(int(tsteps/self.stride + 1))
        for j in range(len(y0)):
            y0[j] = 1.0*xt[j*self.stride]
        return y0

    def generateData2(self, filename=None):
        ''' Generate data and save to file. All particles starting at some x0 
        and propagated for self.timesteps timesteps'''
        if filename == None:
            filename = self.filename
        x0 = np.float32(np.random.randint(self.xmin, self.xmax, self.datasize))
        t0 = self.stride*np.random.randint(0, int(self.timesteps/self.stride), self.datasize)
        num_cores = multiprocessing.cpu_count() 
        results = Parallel(n_jobs=num_cores, verbose = 2)(delayed(self.propagate2)(x, t) for x,t in zip(x0,t0))
        print("Writing to file ...", end="\r")
        f = open(filename, "w")
        for i, result in enumerate(results):
            f.write(str(self.timesteps - t0[i]) + " " + str(result[0]) + " " + str(result[-1]) + "\n")
        f.close()
        print("Percentage finished:", 100, "%    ", end="\r")
    
    def loadData2(self, filename):
        ''' Load data: inputData (x0,t0), targetData u at time tend given x0 at t0=t0'''
        targetData = [None]*(self.datasize)
        data = np.genfromtxt(filename, delimiter=' ')
        inputTime = data[:,0] 
        inputX = data[:,1] 
        inputData = np.reshape(np.column_stack((inputX,inputTime)), [-1,2]) # datasize x 2 corresponds to rows of (t,x)
        for i in range(self.datasize):
            xfinal = data[i, 2]
            targetData[i] = self.terminalCondition(np.array([xfinal]))
        return inputData, targetData