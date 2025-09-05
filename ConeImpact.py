# import libararies 
import numpy as np
import pandas as pd
import scipy
from scipy.integrate import odeint
import math
import pandas as pd
from os.path import dirname, join as pjoin
import matplotlib.pyplot as plt
import re
from scipy.optimize import least_squares
from scipy.optimize import Bounds
from tabulate import tabulate
from scipy import signal
import csv
from scipy.optimize import minimize
from scipy.linalg import eigh  # For generalized eigenvalue problems
import os



# Import external code
from SimulationFunctions import *
from AnalysisFunctions import *

# height is a list of stings 
class coneImpact:
    

    def __init__(self,k, angle, d = None, height = None, model= None, m1 = None, 
                 m2 = None, m = None, mn = None, n = None, datafolder = None, 
                 alpha = None, beta = None):

        # standard inputs 
        self.height     = height  if height is not None else ['025','050','075','100','150']
        self.U0         = np.sqrt(2*9.81*np.array(self.height, dtype = float)/100) 
        self.datafolder = datafolder if datafolder is not None else 'G:\My Drive\PhD\Code\Water entry model\Data\Barrel'
        self.mn  = mn if mn is not None else 0
        self.n   = n if n is not None else 0
        self.alpha = alpha if alpha is not None else 0
        self.beta = beta if beta is not None else 0
        self.model = {}
        if k == 'R':
            k_spring = 0
        elif k == 'S':
            k_spring = 1740.2737546949665
        elif k == 'F':
            k_spring = 8160.32204635801
        else:
            k_spring = k
        damping = d if d is not None else 5
        


        # import data from datafolder
        try:
            self.data = {}
            for h in self.height:
                data_dir = pjoin(datafolder, 'S' + k + angle + h + '.mat')
                self.data.update(scipy.io.loadmat(data_dir))
        except:
        
            pass
        
        if m1 is None:
            # Calculate weights 
            m1 = 522.8/1000
            m_bearing = 34.28
            angle_to_mh = {
                '10': 45.8,
                '12': 43.6,
                '25': 31.2,
                '30': 28.3,
                '40': 32.1,
                '45': 30.1,
                '50': 30.7,
                '60': 30.0,
                '70': 28.0,
                '80': 25.7
            }     
            mh = angle_to_mh.get(angle)
            stiffness_to_ms = { 
                'R': 2.5,
                'S': 1.5,
                'F': 4
            }
            ms = stiffness_to_ms.get(k)

            m2 = (m_bearing+mh+ms)/1000
            m = m1+m2
        else:
            m1 = m1
            m2 = m2
            if m is not None:
                m = m
            else:
                m = m1+m2
        if mn is None:
            mn = 0
            n = 0
        
        self.M = np.concatenate([[m1],mn*np.ones(n),[m2]])

        # Initial values for model fit
        # a = np.linspace(5,90,86)
        # a = np.concatenate(([0], a))
        # Cds = [0.0417, 0.0595, 0.06306, 0.06662, 0.07204, 0.07932, 0.0866, 0.09748, 0.10836, 0.11924, 0.13012, 0.141, 0.15464706,
        #             0.16829412, 0.18194118, 0.19558824, 0.20923529, 0.22288235, 0.23652941, 0.25017647, 0.26530769, 0.28192308, 0.29853846,
        #             0.31515385, 0.33176923, 0.34838462, 0.365, 0.3738, 0.3826, 0.3914, 0.4002, 0.409, 0.4178, 0.4266, 0.4354, 0.4442, 0.453,
        #             0.4618, 0.4706, 0.4794, 0.4882, 0.497, 0.5036, 0.5102, 0.5168, 0.5234, 0.53, 0.5366, 0.5432, 0.5498, 0.5564, 0.563,
        #             0.5696, 0.5762, 0.5828, 0.5894, 0.596, 0.60249, 0.60898, 0.61547, 0.62196, 0.62845, 0.63494, 0.64143, 0.64792, 0.65441,
        #             0.6609, 0.667355, 0.67381, 0.680265, 0.68672, 0.693175, 0.69963, 0.706085, 0.71254, 0.718995, 0.72545, 0.731905, 0.73836,
        #             0.744815, 0.75127, 0.757725, 0.76418, 0.770635, 0.77709, 0.783545, 0.79]

        # # Regression on a and Cds
        # coefficients_Cds = np.polyfit(a, Cds, 3)
        coefficients_Cds = np.array([1.81100050e-07, -8.13339690e-05,  1.49104512e-02, -3.78712427e-02])
        self.Cd = np.polyval(coefficients_Cds, float(angle))


        # remove standard categrories
        try:
            keys_to_remove = ['__header__', '__version__', '__globals__']
            self.data = {key: value for key, value in self.data.items() if key not in keys_to_remove}
            self.tests = list(self.data.keys())
        except:
            pass
        # Add meta data to class
        self.meta = {'m': m , 'R': 0.026, 'g': 9.81, 'dt': 1/4000, 'rho': 997, 'Lb': 0.265, 'm1': m1, 'm2': m2, 'k_spring': k_spring, 'd': damping}
        self.meta.update({'h':  self.meta['R']/np.tan(np.deg2rad(float(angle))),'case': k, 'A': np.pi*self.meta['R']**2, 'beta' : np.deg2rad(float(angle)), 'angle': angle, 'Cds': self.Cd})

        
        # Add model parameters to class
        if model is None:
            self.meta.update({'Lh': (8.88120351e-05*float(angle)**2 -6.45964062e-03*float(angle)+  9.31970340e-01)*self.meta['h'], 'k': .83, 'C1': 0.01}) 
            self.meta.update({'tau':  1})#
        else:
            result = self.getfitedModel()
           

        
        try:
            for test in self.tests:
                self.meta.update({test: {'U0': np.sqrt(2*self.meta['g']*int(test[4:7])/100),
                            'height': int(test[4:7]),
                            'ntest' : int(test[-1])}})
            
                self.meta[test]['tnorm_fac'] = self.meta[test]['U0']/self.meta['h']
                try:
                    self.meta[test]['idx_0'] = min((v, i) for i, v in enumerate(abs(self.data[test][:,0]*self.meta[test]['tnorm_fac'])))[1]
                    self.meta[test]['idx_1'] = min((v, i) for i, v in enumerate(abs(self.data[test][:,0]*self.meta[test]['tnorm_fac']-3)))[1]
                    self.meta[test]['Amax']  = np.max(self.data[test][self.meta[test]['idx_0']:self.meta[test]['idx_1'],1])
                    self.meta[test]['tmax']  = self.data[test][np.where(self.data[test][:,1] == self.meta[test]['Amax'])[0][0],0]
                    self.meta[test]['jerk']  = self.meta[test]['Amax']/self.meta[test]['tmax']
                except:
                    self.meta[test]['idx_0'] = []
                    self.meta[test]['idx_1'] = []
                    self.meta[test]['Amax']  = []
                    self.meta[test]['tmax']  = []
                    self.meta[test]['jerk']  = []
                
            self.mean_for_height('Amax')
            self.mean_for_height('tmax')
            self.mean_for_height('jerk')
        except:
            pass
        
    def getModel(self):
        csv_file_name = 'G:\My Drive\PhD\Code\Water entry model\Python\ModelFit.csv'          # Replace with your CSV file's name
        column_name = self.meta['angle'] # Replace with the name of the column you want to extract
        # Create a list to store the values from the selected column
        column_values = []

        # Open the CSV file and read the data
        with open(csv_file_name, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                if column_name in row:
                    column_values.append(float(row[column_name]))
        
        [self.meta['Lh'], self.meta['k'], self.meta['tau'], self.meta['C1'], self.meta['Cds']] = column_values

        
    
    def getfitedModel(self):

        
        csv_file_name = 'G:\My Drive\PhD\Code\Water entry model\Python\ModelCurves.csv' 

        csv_file = open(csv_file_name, 'r') 
        # Create a CSV reader object with a tab delimiter
        csv_reader = csv.reader(csv_file, delimiter='\t')

        # Read the first row
        coefficients_Lh = next(csv_reader)
        coefficients_Lh = [float(value) for value in coefficients_Lh[0].split(',')]
        coefficients_k = next(csv_reader)
        coefficients_k = [float(value) for value in coefficients_k[0].split(',')]
        coefficients_tau = next(csv_reader)
        coefficients_tau = [float(value) for value in coefficients_tau[0].split(',')]
        # coefficients_tau2 = next(csv_reader)
        # coefficients_tau2 = [float(value) for value in coefficients_tau2[0].split(',')]
        # coefficients_C1 = next(csv_reader)
        # coefficients_C1 = [float(value) for value in coefficients_C1[0].split(',')]

              
        self.meta['Lh'] = np.polyval(coefficients_Lh, float(self.meta['angle']))*self.meta['h']
        self.meta['k'] = np.polyval(coefficients_k, float(self.meta['angle']))

        self.meta['tau'] = np.polyval(coefficients_tau, float(self.meta['angle']))
               
        self.meta['C1'] = 0
    
    def getResluts(self):
        n = self.n
    
        # Example mass (M) and stiffness (K) matrices
        M = np.diag(self.M)
        K = np.zeros((n+2, n+2))
        for i in range(n+2):
            if i == 0: # First mass
                K[i, i] = self.meta['k_spring']
                K[i, i+1] = -self.meta['k_spring']
            elif i == n+1: # Last mass
                K[i, i] = self.meta['k_spring']
                K[i, i-1] = -self.meta['k_spring']
            else: # All other masses
                K[i, i] = 2*self.meta['k_spring']
                K[i, i-1] = -self.meta['k_spring']
                K[i, i+1] = -self.meta['k_spring']
        # Solve the generalized eigenvalue problem
        eigenvalues, eigenvectors = eigh(K, M)
        # Compute the eigenfrequencies
        eigenfrequencies = np.sqrt(eigenvalues) / (2 * np.pi)
        # Compute the maximum forces and times
        F_body_max = np.max(self.model[self.height[0]][:,2*(self.n+2)]*9.81)*self.meta['m1']
        F_head_max = np.max(self.Fwater)
        t_body_max = self.meta['tSim'][np.argmax(self.model[self.height[0]][:,2*(self.n+2)+1])]
        t_head_max = self.meta['tSim'][np.argmax(self.Fwater)]

        d_total = np.min(np.sum(self.model[self.height[0]][:,1:n+2]-self.model[self.height[0]][:,:n+1],axis=1))
        d_max = np.max(abs(self.model[self.height[0]][:,1:n+2]-self.model[self.height[0]][:,:n+1]))
        data = pd.DataFrame({"k": self.meta['k_spring'], "angle" : self.meta['angle'], 
                            "d" : self.meta['d'], "height" : float(self.height[0]), "m1" : self.meta['m1'],
                            "m2" : self.meta['m2'], "m" : self.mn, "n" : self.n, "beta": self.beta,
                            "alpha" : self.alpha, "F_body_max": F_body_max, "F_head_max": F_head_max,
                            "transfer": F_body_max/F_head_max, "d_total": d_total, "d_max":d_max,
                            "t_body_max": t_body_max, "t_head_max": t_head_max,
                            "eigenfrequencies0": eigenfrequencies[1]} , index=[0]) 
        self.results = data

    def add2Results(self, file_root, file_name=None):
        file_name = file_name if file_name is not None else 'Results.csv'
        file_path = os.path.join(file_root, file_name)
        if os.path.exists(file_path):
            # Append the DataFrame to the CSV file
            self.results.to_csv(file_path, mode="a", index=False, header=False)
        else:
            self.results.to_csv(file_path, index=False)
      
    ###############################################################################################################################
    ########                                       ANALYIS                                                         ################        
    ###############################################################################################################################

    def impulse(self,tend, tstart = None, type = None):
        impulse_(self, tend, tstart, type)


    def mean_for_height(self,name):
        mean_for_height_(self,name)

    
    def CdPeak(self, type = None):
        Cmax_, tmax_, Cmax_model_, tmax_model_ = CdPeak_(self, type)    
       
        return Cmax_, tmax_, Cmax_model_, tmax_model_
    
    
    def CdPeakSpring(self):
        Cmax_, tmax_, Cmax_model_, tmax_model_ = CdPeakSpring_(self)
        return Cmax_, tmax_, Cmax_model_, tmax_model_
    


    def GainPhase(self):
        freq, gain, phase = GainPhase_(self)
        return freq, gain, phase
        


    ###############################################################################################################################
    ########                                       SIMULATION/MODEL                                                ################        
    ###############################################################################################################################


    def Sim(self, tend = None):
        Sim_(self, tend)
        
    def SimSpring(self, tend= None, d =None, k = None):
        SimSpring_(self, tend, d, k)

    def BaldwinRigid(self, y,t):
        dz, dzdt = BaldwinRigid_(self, y, t)
        return [dz,dzdt]

    def BaldwinSpring(self, y,t):
        dz1, dz1dt, dz2, dz2dt  = BaldwinSpring_(self, y, t)
        return [dz1, dz1dt, dz2, dz2dt]
    
    def addedMass(self,z):
        m = addedMass_(self,z)
        return m

    def Fb(self,z):
        Fb = Fb_(self,z)
        return Fb

    def Fd(self, z, dz):
        Fd = Fd_(self,z,dz)      
        return Fd


 ###############################################################################################################################
 ########                                       PLOTS                                                           ################        
 ###############################################################################################################################    

    # Time acceleration plots 
    def plot(self, expIdx=None, heights=None, type = None, color = None,style = None):
        expIdx = expIdx if expIdx is not None else list(range(1, 7))
        heights = heights if heights is not None else self.height
        style = style if style is not None else '-'
        categories = list(self.data.keys())
        for test in categories: 
            if test[4:7] in heights:
                if int(test[-1]) in expIdx:
                    if type is None:
                        if color is None:
                            plt.plot(self.data[test][:,0],9.81*self.data[test][:,1]*self.meta['m'], label = test)
                        else:
                            plt.plot(self.data[test][:,0],9.81*self.data[test][:,1]*self.meta['m'], label = test, color = color, linestyle = style)
                    else:
                        if color is None:
                            plt.plot(self.data[test][:,0],9.81*self.data[test][:,1]*self.meta['m1'], label = test)
                        else:
                            plt.plot(self.data[test][:,0],9.81*self.data[test][:,1]*self.meta['m1'], label = test, color = color, linestyle = style)
        plt.xlabel('t (s)')
        plt.ylabel('F (N)')
        # plt.title(r'$\beta$ ' + str(self.tests[1][2:4]) + r'$^o$')
        
    def plotModel(self,type=None, heights=None, plotHead = None, color = None, style = None):
        heights = heights if heights is not None else self.height
        style = style if style is not None else '-'
        if type is not None:
            for height in heights: 
                if color is None:
                    plt.plot(self.meta['tSim'],9.81*self.model[height][:,2]*self.meta['m1'], color = 'red', label = 'Model ' + r'm_1 ' + self.meta['angle'] + r'$^o$ ' +  height ) 
                else:
                    plt.plot(self.meta['tSim'],9.81*self.model[height][:,2]*self.meta['m1'], color = color, linestyle = style, label = 'Model ' + r'm_1 ' + self.meta['angle'] + r'$^o$ ' +  height )
                if plotHead == True:
                    plt.plot(self.meta['tSim'],9.81*self.model[height][:,-1]*self.meta['m2'] + self.meta['k_spring']*(self.model[height][:,0]-self.model[height][:,3]), color = 'red', label = 'Model ' + r'm_2' + self.meta['angle'] + r'$^o$ ' +  height )
        else:
            for height in heights: 
                plt.plot(self.meta['tSim'],9.81*self.model[height][:,-1]*self.meta['m'], color = 'red')              
        plt.xlabel('t (s)')
        plt.ylabel('F (N)')
        # plt.title(r'$\beta$ = ' + str(self.tests[1][2:4]) + r'$^o$')

    def plotNModel(self, type=None, heights=None, plotHead = None, color = None, style = None):
        N = self.n
        # plot head
        plt.plot(self.meta['tSim'],(self.model[self.height[0]][:,2*(N+2)+1]), color = 'red', label = 'Head')
        # plot body
        plt.plot(self.meta['tSim'],self.model[self.height[0]][:,-1], color = 'blue', label = 'Model ' + r'm_2' + self.meta['angle'] + r'$^o$ ' +  self.height[0])
        # plot vertebra
        for i in range(N):
            plt.plot(self.meta['tSim'],self.model[self.height[0]][:,2*(N+2)+1+i], label = 'Model ' + r'm_n ' + self.meta['angle'] + r'$^o$ ' +  self.height[0])
        plt.xlabel('t (s)')
        plt.ylabel('F (N)')

    # time acceleration average of test data
    def plotAvg(self, type = None, heights=None):
        heights = heights if heights is not None else self.height
        categories = list(self.data.keys())
        for heigth in self.height:
            tstart = []
            tend   = []
            name_tests = []

            for test in categories:
                if test[4:7] == heigth:
                    tstart.append(self.data[test][0,0])
                    tend.append(self.data[test][-1,0])
                    name_tests.append(test)
                else:
                    pass
            common_time = np.linspace(max(tstart),min(tend),10000)
            intr_curves = []
            for test in name_tests:
                intr_curves.append((np.interp(common_time, self.data[test][:,0], self.data[test][:,1])))
            average_curve = np.mean(intr_curves, axis=0)
            max_curve     = np.max(intr_curves, axis=0)
            min_curve     = np.min(intr_curves, axis =0)
            # plt.plot(common_time, average_curve) 
            if type is None:
               plt.fill_between(common_time, min_curve*self.meta['m'],max_curve*self.meta['m'], color='blue', alpha=0.2, label='Range test data')
            else:
                plt.fill_between(common_time, min_curve*self.meta['m1'], max_curve*self.meta['m1'], color='blue', alpha=0.2, label='Range test data')
            plt.xlabel('t (s)')
            plt.ylabel('F (N)')
            # plt.title(r'$\beta$ = ' + str(self.tests[1][2:4]) + r'$^o$')
                 

    # Plots with normalized on coefficient of drag
    def plotCd(self,expIdx=None,heights= None, type = None, color = None, style = None):
        style = style if style is not None else '-'
        expIdx = expIdx if expIdx is not None else list(range(1, 7))
        tests = list(self.data.keys())
        heights = heights if heights is not None else self.height
        A = self.meta['A']
        for test in tests:
            if test[4:7] in heights:
                if int(test[-1]) in expIdx:
                    U0  = self.meta[test]['U0']
                    if type is None:
                        if color is None:
                            plt.plot(self.data[test][:,0]*self.meta[test]['tnorm_fac'],9.81*self.data[test][:,1]*self.meta['m']*2/(self.meta['rho']*A*U0**2), label= test)
                        else:
                            plt.plot(self.data[test][:,0]*self.meta[test]['tnorm_fac'],9.81*self.data[test][:,1]*self.meta['m']*2/(self.meta['rho']*A*U0**2), label= test, color = color, linestyle = style)
                    else:
                        if color is None:
                            plt.plot(self.data[test][:,0]*self.meta[test]['tnorm_fac'],9.81*self.data[test][:,1]*self.meta['m1']*2/(self.meta['rho']*A*U0**2), label= test)
                        else:
                            plt.plot(self.data[test][:,0]*self.meta[test]['tnorm_fac'],9.81*self.data[test][:,1]*self.meta['m1']*2/(self.meta['rho']*A*U0**2), label= test, color = color, linestyle = style)
                    # plt.legend(loc='upper right', title='Test')
        plt.axvline(x=1, color='k', linestyle='--', label='Full cone sub.')
        # plt.title(r'Cd $\beta$ = ' + str(self.tests[1][2:4]) + r'$^o$')
        plt.xlabel('t/(h/V)')
        plt.ylabel(r'Cd = F/(($\rho$AV$^2$)/2)')
    

    
    def plotModelCd(self, heights = None, type = None):
        heights = heights if heights is not None else self.height
        if type is not None:
            for height in heights: 
                U0 =np.sqrt(2*self.meta['g']*float(height)/100)
                plt.plot(self.meta['tSim']/(self.meta['h']/U0),9.81*self.model[height][:,-2]*self.meta['m1']*2/(self.meta['rho']*self.meta['A']*U0**2), color = 'blue')
                plt.plot(self.meta['tSim']/(self.meta['h']/U0),9.81*self.model[height][:,-1]*self.meta['m2']*2/(self.meta['rho']*self.meta['A']*U0**2), color = 'red')
        else:
            for height in heights: 
                U0 =np.sqrt(2*self.meta['g']*float(height)/100)
                plt.plot(self.meta['tSim']/(self.meta['h']/U0),9.81*self.model[height][:,2]*self.meta['m']*2/(self.meta['rho']*self.meta['A']*U0**2))

        plt.axvline(x=1, color='k', linestyle='--', label='Full cone sub.')
        # plt.title(r'Cd $\beta$ = ' + str(self.tests[1][2:4]) + r'$^o$')
        plt.xlabel('t/(h/V)')
        plt.ylabel(r'Cd = F/(($\rho$AV$^2$)/2)')
        
 ###############################################################################################################################
 ########                                       MODEL FIT                                                       ################        
 ###############################################################################################################################     

    def fitModel(self, tend= None):

        categories = list(self.data.keys())
        self.meta['tSim'] = np.linspace(0,tend,2001) if tend is not None else self.meta['tspan']

        intrpl_data = {}
        for test in categories:
            intrpl_data[test] = np.interp(self.meta['tSim'], self.data[test][:,0], self.data[test][:,1])
               
        initial_params = [ self.meta['k'], self.meta['tau'], self.meta['Lh']]#,,self.meta['Cds'] self.meta['C1'],

        result = minimize(self.residuals, initial_params, args=(self.meta['tSim'], intrpl_data), method='Nelder-Mead') #, self.meta['Cds'], self.meta['C1']
        #result = least_squares(self.residuals, initial_params , bounds=([0,0,0],[np.inf,np.inf,np.inf]), args=(self.meta['tSim'], intrpl_data)) #
    

        self.meta.update({ 'k': result.x[0], 'tau': result.x[1],'Lh': result.x[2]})#, 'Cds': result.x[4], 'C1': result.x[2],

        return result.x

    def residuals(self, params, t, data):
        total_residuals = []
        for height in self.height:
            initial_conditions = [0, np.sqrt(2*self.meta['g']*(float(height)/100))]
            y_model = odeint(self.my_model, initial_conditions, t, args=(params,))
            
            acceleration = []
            for i in range(len(y_model[:,0])):
                acceleration_i = self.my_model([y_model[i,0],y_model[i,1]],0, params)
                acceleration.append((-acceleration_i[1]/ 9.81)+1)
            acceleration = np.array(acceleration)
            y_model = np.column_stack((y_model, acceleration))

            # for datasets in data
            for dataset in data:
                if dataset[4:7] == height:
                    residuals = (y_model[:,2] - data[dataset])**2
                    # residuals = y_model[:,2] - data[dataset]
                    total_residuals.extend(residuals.flatten())

        #return np.array(total_residuals)
        return np.sum(np.array(total_residuals))

    def my_model(self, y, t ,params):
        z, dz  = y

        
        k  = params[0]
        tau = params[1]
        #C1  = params[2]
        Cds = self.meta['Cds']
        Lh = params[2]#self.meta['Lh']

        m = np.where( z < Lh,
                    k*self.meta['rho']*np.tan(self.meta['beta'])**3*z**3,
                    k*self.meta['rho']*np.tan(self.meta['beta'])**3*Lh**3 -(3*self.meta['k']*self.meta['rho']*np.tan(self.meta['beta'])**3*Lh**2)*(2*(Lh*np.exp(tau*(-np.sqrt((z/Lh) -1)))*(tau*np.sqrt((z/Lh)-1)+1)-Lh))/tau**2  #(Lh*(np.exp(tau*(Lh-z)/Lh)-1)) / tau
                    #+ 3*k*self.meta['rho']*np.tan(self.meta['beta'])**3*Lh**2/(1+C1)*((Lh-Lh*np.exp(tau*(1-(z/Lh))))/tau  + C1*(self.meta['R']**2*(Lh-z)/(-Lh+self.meta['R']+z)**2)              )#3*k*self.meta['rho']*np.tan(self.meta['beta'])**3*Lh**2* ((1 - np.exp(tau-((tau*z)/Lh)))/tau + (C1*self.meta['R']*(z-Lh))/(z-Lh+self.meta['R']))/(1+C1)
        )

        Fb = self.Fb(z)

        Cdsi = ((abs(Lh+z)-abs(Lh-z))/(2*Lh))**2*Cds
        K =  k + Cdsi*np.pi/(6*np.tan(self.meta['beta']))
        Cd = (6*K*np.tan(self.meta['beta']))/np.pi
        
        At = np.tan(self.meta['beta'])**2*z**2*np.pi
        ALh  = np.tan(self.meta['beta'])**2*Lh**2*np.pi


        Fd = np.where ( z < Lh,
                    self.meta['rho']/2*Cd*At*dz**2,
                    self.meta['rho']/2*Cds*ALh*dz**2 + (3*k*self.meta['rho']*np.tan(self.meta['beta'])**3*Lh**2*dz**2)*np.exp(-tau*((z-Lh)/Lh)**0.5)# * ((np.exp(-tau*((z-Lh)/Lh)) + C1 /(1+(z-Lh)/self.meta['R'])**2)/ (C1+1)) 
        )   

        dzdt = (-Fb+ self.meta['m']*self.meta['g'] - Fd)/(self.meta['m']+m)
        
        return [dz,dzdt]

        
