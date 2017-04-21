import random
import numpy
import math


class Cuckoo():
    def __init__(self, NP, N_Gen, pa, Lower, Upper, function):
        self.D = len(Lower)  # dimension
        self.NP = NP  # population size
        self.N_Gen = N_Gen  # generations

        self.Lower = Lower  # lower bound
        self.Upper = Upper  # upper bound

        self.f_min = 0.0  # minimum fitness
        self.pa = pa  # alpha

        self.Sol =    [[0 for i in range(self.D)] for j in range(self.NP)]  # population of solutions
        self.newSol = [[0 for i in range(self.D)] for j in range(self.NP)]  # population of solutions
        self.Fitness = [0] * self.NP  # fitness
        self.best = [0] * self.D  # best solution
        self.Fun = function

    def init_ff(self):

        for i in range(self.NP):
            for j in range(self.D):
                rnd = random.uniform(0, 1)
                self.Sol[i][j] = self.Lower[j] + (self.Upper[j] - self.Lower[j]) * numpy.random.uniform(0, 1)
            self.Fitness[i] = self.Fun(self.Sol[i])

    def findRange(self):
        for i in range(len(self.newSol)):
            for j in range(len(self.newSol[i])):
                if self.newSol[i][j] < self.Lower[j]: self.newSol[i][j] = self.Lower[j]
                if self.newSol[i][j] > self.Upper[j]: self.newSol[i][j] = self.Upper[j]

    def get_best_nest(self):
        #% Evaluating all new solutions
        for j in range(self.NP):
            fnew=self.Fun(self.newSol[j])
            if fnew<self.Fitness[j]:
               self.Fitness[j]=fnew
               self.Sol[j]=[self.newSol[j][k] for k in range(self.D)]
            #% Find the current best
            if fnew<self.f_min:
               self.f_min=fnew
               self.best=[self.newSol[j][k] for k in range(self.D)]


    def get_cuckoos(self):
        # levy flights
        n = self.NP

        # levy exp and coef
        beta=3.0/2
        sigma = (math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)

        for i in range(n):
            s = [self.Sol[i][j] for j in range(self.D)]
            # levy flights by mantegna's algorithm
            u = [numpy.random.normal()*sigma for j in range(self.D)]
            v = [numpy.random.normal() for j in range(self.D)]
            step = [u[j]/(math.fabs(v[j])**(1/beta)) for j in range(self.D)]

            stepsize = [0.01*step[j]*(s[j]-self.best[j]) for j in range(self.D)]
            s=[s[j]+stepsize[j]*numpy.random.normal() for j in range(self.D)]
            for j in range(self.D):
                if s[j]<self.Lower[j]:
                    self.newSol[i][j] = self.Lower[j]
                elif s[j]>self.Upper[j]:
                    self.newSol[i][j] = self.Upper[j]
                else:
                    self.newSol[i][j] = s[j]

    def empty_nest(self):
        n = self.NP
#        new_nest = [[self.Sol[i][j] for j in range(self.D)] for i in range(n)]
        for i in range(n):
            K = [numpy.random.uniform(0,1)>self.pa  for j in range(self.NP)]
            rand = numpy.random.uniform(0,1)

            perm=numpy.random.permutation(n)

            ind1=perm[0]
            ind2=perm[1]

            nest1 = [self.Sol[ind1][j] for j in range(self.D)]
            nest2 = [self.Sol[ind2][j] for j in range(self.D)]
            stepsize = [rand*(nest1[j]-nest2[j]) for j in range(self.D)]
            #self.newSol=[self.Sol[i][j] for j in range(self.D)]
            for j in range(self.D):
                if K[j]:
                    self.newSol[i][j] = self.Sol[i][j] + stepsize[j]
                    if self.newSol[i][j]>self.Upper[j]:
                        self.newSol[i][j] = self.Upper[j]
                    elif self.newSol[i][j]<self.Lower[j]:
                        self.newSol[i][j]=self.Lower[j]


    def move_cc(self):
        self.init_ff()
        t=0
        while t <self.N_Gen:
            self.get_cuckoos()
            self.get_best_nest()
            t = t+self.NP
            self.empty_nest()
            self.get_best_nest()
            t = t+self.NP
#            print self.f_min

class BBatAlgorithm():

    # no need to change
    def __init__(self, D, NP, N_Gen, A, r, Qmin,
	Qmax, function):
        self.D = D  #dimension
        self.NP = NP  #population size
        self.N_Gen = N_Gen  #generations
        self.A = A  #loudness
        self.r = r  #pulse rate
        self.Qmin = Qmin  #frequency min
        self.Qmax = Qmax  #frequency max

        self.f_min = 0.0  #minimum fitness

        self.Lb = [0] * self.D  #lower bound
        self.Ub = [0] * self.D  #upper bound
        self.Q = [0] * self.NP  #frequency

        self.v = [[0 for i in range(self.D)] for j in range(self.NP)]  #velocity
        self.Sol = [[False for i in range(self.D)] for j in range(self.NP)]  #population of solutions
        self.Fitness = [0] * self.NP  #fitness
        self.best = [0] * self.D  #best solution
        self.Fun = function

    # the best individu is defined by minimize function
    def best_bat(self):
        j = 0
        for i in range(self.NP):
            if self.Fitness[i] < self.Fitness[j]:
                j = i
        for i in range(self.D):
            self.best[i] = self.Sol[j][i]
        self.f_min = self.Fitness[j]

    # need to modify in representation of individu to binary
    # solve
    def init_bat(self):
        for i in range(self.NP):
            self.Q[i] = 0
            for j in range(self.D):
                rnd = numpy.random.uniform(0, 1)
                self.v[i][j] = 0.0
                if (rnd<=numpy.random.uniform(0, 1)):
                    self.Sol[i][j] = 0
                else :
                    self.Sol[i][j] = 1
                #self.Sol[i][j] = self.Lb[j] + (self.Ub[j] - self.Lb[j]) * rnd

            self.Fitness[i] = self.Fun(self.Sol[i])
        self.best_bat()

    # need to modify in transfer function, done
    def move_bat(self):
        goodInitiation = False
        while not goodInitiation:
            self.init_bat()
            testGoodness = [0 for i in range(self.D)]
            for ii in self.Sol:
                for idx, jj in enumerate(ii):
                    if jj == 1: testGoodness[idx] = 1
            goodInitiation = numpy.sum(testGoodness) == self.D


        for t in range(self.N_Gen):
            print("Generation : ", t)
            for i in range(self.NP):
                tempSol = [self.Sol[i][k] for k in range(self.D)]
                for j in range(self.D):
                    rnd = numpy.random.uniform(0, 1)
                    self.Q[i] = self.Qmin + (self.Qmax - self.Qmin) * rnd
                    self.v[i][j] = self.v[i][j] + (self.Sol[i][j] -
                                                   self.best[j]) * self.Q[i]
                    rnd = numpy.random.uniform(0,1)

                    # Equation 9 in the paper
                    V_shaped_transfer_function = math.fabs((2/math.pi)*
                                                           math.atan((math.pi/2)*
                                                                     self.v[i][j]))

                    if rnd < V_shaped_transfer_function:
                        if tempSol[j]:
                           tempSol[j] = 0
                        else:
                           tempSol[j] = 1

                    rnd = numpy.random.uniform(0,1)

                    if rnd > self.r:
                        tempSol[j]=self.best[j]

                Fnew = self.Fun(tempSol)

                rnd = numpy.random.uniform(0,1)

                if (Fnew <= self.Fitness[i]) and (rnd < self.A):
                    for j in range(self.D):
                        self.Sol[i][j] = tempSol[j]
                    self.Fitness[i] = Fnew

                if Fnew <= self.f_min:
                    for j in range(self.D):
                        self.best[j] = tempSol[j]
                    self.f_min = Fnew
