
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

###############################################################################################################################
########                                       SIMULATION/MODEL                                                ################        
###############################################################################################################################


def Sim_(self, tend = None):
    self.meta['tSim'] = np.linspace(0,tend,10001) if tend is not None else np.linspace(0,0.07,10001)
    
    for height in self.height:
        U0    = np.sqrt(2*self.meta['g']*int(height)/100) 
        y0    = [0, U0]

        self.model[height] = odeint(self.BaldwinRigid,y0,self.meta['tSim'])
        
        acceleration = []
        for i in range(len(self.model[height][:,0])):
            acceleration_i = self.BaldwinRigid([self.model[height][i,0],self.model[height][i,1]],0)
            acceleration.append((-acceleration_i[1]/ 9.81)+1)
        acceleration = np.array(acceleration)
        self.model[height] = np.column_stack((self.model[height], acceleration))
  

def SimSpring_(self, tend = None, d = None, k = None):
    
    # Adjust Damping ratio if needed
    self.meta['d'] = d  if d is not None else self.meta['d']
    self.meta['k_spring'] = k  if k is not None else self.meta['k_spring']

    self.meta['tSim'] = np.linspace(0,tend,10000, endpoint= False) if tend is not None else np.linspace(0,0.07,10000,endpoint= False)
    
    # if self.n == 0:
    #     for height in self.height:
    #         U0    = np.sqrt(2*self.meta['g']*int(height)/100)
    #         y0    = [0, U0, 0, U0]
    #         self.model[height] = odeint(self.BaldwinSpring,y0,self.meta['tSim'])
            
    #         acceleration = []
    #         for i in range(len(self.model[height][:,0])):
    #             acceleration_i = self.BaldwinSpring([self.model[height][i,0],self.model[height][i,1],self.model[height][i,2],self.model[height][i,3]],0)
    #             acceleration.append([(-a / self.meta['g']) + 1 for a in acceleration_i])
    #         acceleration = np.array(acceleration)
    #         self.model[height] = np.column_stack((self.model[height][:,0:2], acceleration[:,1], self.model[height][:,2:4], acceleration[:,3]))
# else:
    n = self.n
    height = self.height[0]
    U0    = self.U0[0]
    y0 =  [0,0,U0,U0]#np.concatenate((np.zeros(n+2), U0*np.ones(n+2)))
    sol = solve_ivp(lambda t, y: equation_of_motion_over_time(t,y,self), (self.meta['tSim'][0],self.meta['tSim'][-1]), y0, t_eval=self.meta['tSim'])

    # Extract the results
    z_sol = sol.y[:len(sol.y)//2]
    dz_sol = sol.y[len(sol.y)//2:]
    # Calculate ddz_sol
    ddz_sol = np.array([EOM_N_elements_(self, z, dz) for z, dz in zip(z_sol.T, dz_sol.T)])
    self.Fwater = np.array([Fb_(self,z[-1]) + Fd_(self, z[-1], dz[-1]) for z, dz in zip(z_sol.T, dz_sol.T)])
    self.model[height] = np.column_stack((z_sol.T, dz_sol.T, (-ddz_sol/self.meta['g'])+1))

def equation_of_motion_over_time(t,y,self):
        z = y[:len(y)//2]
        dz = y[len(y)//2:]
        ddz = EOM_N_elements_(self,z, dz)
        return np.concatenate((dz, ddz))


def BaldwinRigid_(self, y,t):
    # Rigid baldwin model
    # Pre-peak added mass increases qubic
    # After peak(Lh) an exponential is chosen to capture the further increase in added mass 
    z, dz  = y

    m = self.addedMass(z)
    Fb = self.Fb(z)
    Fd = self.Fd(z, dz)
    
    dzdt = (-Fb+ self.meta['m']*self.meta['g'] - Fd)/(self.meta['m']+m)

    return [dz,dzdt]

def BaldwinSpring_(self, y,t):
    # Rigid baldwin model
    # Pre-peak added mass increases qubic
    # After peak(Lh) an exponential is chosen to capture the further increase in added mass 
    z1, dz1, z2, dz2  = y
    
    m = self.addedMass(z1)
    Fb = self.Fb(z1)
    Fd = self.Fd(z1, dz1)
    d = self.meta['d']

    dz1dt = (-self.meta['k_spring']*(z1-z2) - d*(dz1-dz2)+self.meta['m1']*self.meta['g'])/self.meta['m1']
    dz2dt = (-Fb+ self.meta['m2']*self.meta['g'] - Fd + self.meta['k_spring']*(z1-z2) + d*(dz1-dz2))/(self.meta['m2']+m)

    return [dz1, dz1dt, dz2, dz2dt]

def EOM_N_elements_(self,z, dz):
    n = self.n
    ddz = np.zeros(n+2)
    g = self.meta['g']
    m = self.M
    alpha = np.ones(n+1)*self.alpha
    beta = np.ones(n+1)*self.beta
    k = np.ones(n+1)*self.meta['k_spring']
    c = np.ones(n+1)*self.meta['d']
      
    # Calculate water forces
    F_water = Fb_(self,z[-1]) + Fd_(self, z[-1], dz[-1])
    for i in range(n+2):
                    
        if i == 0: # First mass Body
            dx = z[i] - z[i+1]
            ddx =  dz[i] - dz[i+1]
            ddz[i] = (-k[i]*dx - alpha[i]*dx**2 - beta[i]*dx**3 - c[i]*ddx  ) / m[i] + g
        elif i == n+1: # Last mass cone
            dx  =    z[i] - z[i-1]
            ddx =   dz[i] - dz[i-1]
            ddz[i] = (-k[i-1]*dx - alpha[i-1]*dx**2 - beta[i-1]*dx**3 - c[i-1]*ddx + g*m[i] - F_water) / (m[i] + addedMass_(self,z[i]))
        else: # All other m
            dx1 = z[i] - z[i-1]
            dx2 = z[i] - z[i+1]
            ddx1 = dz[i] - dz[i-1]
            ddx2 = dz[i] - dz[i+1]
            ddz[i] = (-k[i-1]*dx1 -k[i]*dx2 -alpha[i-1]*dx1**2 -alpha[i]*dx2**2 -beta[i-1]*dx1**3 -beta[i]*dx2**3 -c[i-1]*ddx1 -c[i]*ddx2 ) / m[i] + g
    return ddz

def addedMass_(self,z):
    m = np.where( z < self.meta['Lh'],
                    self.meta['k']*self.meta['rho']*np.tan(self.meta['beta'])**3*z**3,
                    self.meta['k']*self.meta['rho']*np.tan(self.meta['beta'])**3*self.meta['Lh']**3 -(3*self.meta['k']*self.meta['rho']*np.tan(self.meta['beta'])**3*self.meta['Lh']**2)*(2*(self.meta['Lh']*np.exp(self.meta['tau']*(-np.sqrt((z/self.meta['Lh']) -1)))*(self.meta['tau']*np.sqrt((z/self.meta['Lh'])-1)+1)-self.meta['Lh']))/self.meta['tau']**2 #(self.meta['Lh']*(np.exp(self.meta['tau']*(self.meta['Lh']-z)/self.meta['Lh'])-1)) / self.meta['tau'] #3*self.meta['k']*self.meta['rho']*np.tan(self.meta['beta'])**3*self.meta['Lh']**2* ((self.meta['Lh'] - self.meta['Lh']* np.exp(self.meta['tau']-((self.meta['tau']*z)/self.meta['Lh'])))/self.meta['tau'] + (self.meta['C1']*self.meta['R']*(z-self.meta['Lh']))/(z-self.meta['Lh']+self.meta['R']))/(1+self.meta['C1'])
    )
    # self.meta['k']*self.meta['rho']*np.tan(self.meta['beta'])**3*self.meta['Lh']**3 + (((3*self.meta['k']*self.meta['rho']*np.tan(self.meta['beta'])**3*self.meta['Lh']**2* (self.meta['Lh'] - self.meta['Lh']*np.exp(self.meta['tau']-((self.meta['tau']*z)/self.meta['Lh'])))/self.meta['tau'])+ self.meta['C1'] * self.meta['R']*self.meta['tau']*(z-self.meta['Lh'])/(-self.meta['Lh']+self.meta['R']+z)))/((1+self.meta['C1'])))
    return m

def Fb_(self,z):
    Fb = np.where(
        z <= self.meta['h'],
        self.meta['A'] * self.meta['rho'] * self.meta['g'] * z/3,
        np.where(
            (self.meta['h'] < z) & (z < self.meta['h'] + self.meta['Lb']),
            self.meta['A'] * self.meta['rho'] * self.meta['g'] * (self.meta['h'] / 3 + (z - self.meta['h'])),
            self.meta['A'] * self.meta['rho'] * self.meta['g'] * (self.meta['h'] / 3 + self.meta['Lb'])
        )
    )
    return Fb
    # Fb = np.where( z < (self.meta['Lb']+self.meta['h']),
    #                 self.meta['A']*self.meta['rho']*self.meta['g']*(z-(2/3)*((abs(self.meta['h']+z)-abs(self.meta['h']-z))/2)),
    #                 self.meta['A']*self.meta['rho']*self.meta['g']*((self.meta['Lb']+self.meta['h'])-(2/3)*((abs(self.meta['h']+z)-abs(self.meta['h']-z))/2)) )
    # return Fb


def Fd_(self, z, dz):
    Cdsi = ((abs(self.meta['Lh']+z)-abs(self.meta['Lh']-z))/(2*self.meta['Lh']))**2*self.meta['Cds']
    K =  self.meta['k'] + Cdsi*np.pi/(6*np.tan(self.meta['beta']))
    Cd = (6*K*np.tan(self.meta['beta']))/np.pi
    
    At = np.tan(self.meta['beta'])**2*z**2*np.pi
    ALh  = np.tan(self.meta['beta'])**2*self.meta['Lh']**2*np.pi

    Fd = np.where ( z < self.meta['Lh'],
                    self.meta['rho']/2*Cd*At*dz**2,
                    self.meta['rho']/2*self.meta['Cds']*ALh*dz**2 +  (3*self.meta['k']*self.meta['rho']*np.tan(self.meta['beta'])**3*self.meta['Lh']**2*dz**2)*np.exp(-self.meta['tau']*((z-self.meta['Lh'])/self.meta['Lh'])**0.5)    # dm/ds e and **2 (3*self.meta['k']*self.meta['rho']*np.tan(self.meta['beta'])**3*self.meta['Lh']**2*dz**2) * ((np.exp(-self.meta['tau']*((z-self.meta['Lh'])/self.meta['Lh'])) + self.meta['C1']/(1+((z-self.meta['Lh'])/self.meta['R']))**2)/ (self.meta['C1']+1)) 
    )
    
    return Fd