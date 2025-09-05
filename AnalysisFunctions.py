###############################################################################################################################
########                                       ANALYIS                                                         ################        
###############################################################################################################################

import numpy as np
import math

def impulse_(self,tend, tstart = None, type = None):
    if type is None:
        for test in self.tests:
            idx_start   = self.meta[test]['idx_0'] if tstart == None else min((v, i) for i, v in enumerate(abs(self.data[test][:,0]-tstart)))[1] 
            idx_end     = min((v, i) for i, v in enumerate(abs(self.data[test][:,0]-tend)))[1] 
            if self.meta['k_spring'] == 0:
                self.meta[test]['impulse'] = np.sum(self.meta['m1'] * self.data[test][idx_start:idx_end,1]*self.meta['dt'])
            else:
                self.meta[test]['impulse'] = np.sum(self.meta['m1'] * self.data[test][idx_start:idx_end,1]*self.meta['dt'])
        self.mean_for_height('impulse')
    elif type == 'Model':
        self.meta['impulse'] = []
        for height in self.height:
            if (self.meta['tSim'][-1]-tend) < 0:
                    print('tend exceeds max. val. in tSim')
            else:
                idx_end     = np.argmin(abs(self.meta['tSim']-tend))
                self.meta['impulse'].append(np.sum(self.meta['m1'] * self.model[height][0:idx_end,2]*(self.meta['tSim'][1]-self.meta['tSim'][0])))
                


def mean_for_height_(self, name):
    name_mean = str(name + '_mean')
    self.meta.update({name_mean:[]})
    for height in self.height:
            i = 0
            x = 0
            for test in self.tests:
                if self.meta[test]['height'] == int(height):
                        i = i+1
                        x = x + self.meta[test][name] 
            self.meta.update({ name_mean: self.meta[name_mean] + [x/i]}) 

def CdPeak_(self, type = None):
    Cmax = []
    tmax = []
    try: 
        tests = list(self.data.keys())
        A = self.meta['A']
        for test in tests: 
            U0  = self.meta[test]['U0']
            idx1 = np.argmin(abs(self.data[test][:,0]*self.meta[test]['tnorm_fac']))
            idx2 = np.argmin(abs(self.data[test][:,0]*self.meta[test]['tnorm_fac']-2)) if type is None else np.argmin(abs(self.data[test][:,0]*self.meta[test]['tnorm_fac']-2))             
            maxIdx = np.argmax(self.data[test][idx1:idx2,1]*self.meta['m']*2/(self.meta['rho']*A*U0**2))
            if (idx2-idx1)-maxIdx < 0.1*(idx2-idx1):
                idxLh = np.argmin(abs(self.data[test][:,0]*self.meta[test]['tnorm_fac']-(self.meta['Lh']/self.meta['h'])))
                Cmax.append(self.meta['g']*self.data[test][idxLh,1]*self.meta['m']*2/(self.meta['rho']*A*U0**2))
                tmax.append(self.data[test][idxLh,0]*self.meta[test]['tnorm_fac'])
            else:
                Cmax.append(self.meta['g']*self.data[test][maxIdx+idx1,1]*self.meta['m']*2/(self.meta['rho']*A*U0**2))
                tmax.append(self.data[test][maxIdx+idx1,0]*self.meta[test]['tnorm_fac'])
        
        Cmax_ = np.mean(Cmax)
        tmax_ = np.mean(tmax)
    except:
        Cmax_ = []
        tmax_ = []
        
    
    Cmax_model = []
    tmax_model = []
    try:
        for height in self.height: 
            U0 = np.sqrt(2*self.meta['g']*int(height)/100)
            tidx = np.argmax(self.model[height][:,-1]) 
            tlen = len(self.model[height][:,-1])
            
            if type is not None:
                Cmax_model.append(self.meta['g']*self.model[height][tidx,-1]*self.meta['m1']*2/(self.meta['rho']*self.meta['A']*U0**2))
            else:
                if tlen-tidx < 0.1*tlen:
                    #idxLh = np.argmin(abs(self.meta['tSim']*(U0/self.meta['h'])-(self.meta['Lh']/self.meta['h'])))
                    #tmax_model.append(self.meta['tSim'][idxLh]*(U0/self.meta['h']))
                    Cmax_model.append(self.meta['g']*np.max(self.model[height][:,-1])*self.meta['m']*2/(self.meta['rho']*self.meta['A']*U0**2))
                    tmax_model.append(self.meta['tSim'][np.argmax(self.model[height][:,-1])]*(U0/self.meta['h']))
                else:
                    # tmax_model.append(self.meta['tSim'][tidx]*(U0/self.meta['h']))
                    # Cmax_model.append(self.meta['g']*self.model[height][tidx,-1]*self.meta['m']*2/(self.meta['rho']*self.meta['A']*U0**2))
                    tmax_model.append(self.meta['tSim'][np.argmax(self.model[height][:,-1])]*(U0/self.meta['h']))
                    Cmax_model.append(self.meta['g']*np.max(self.model[height][:,-1])*self.meta['m']*2/(self.meta['rho']*self.meta['A']*U0**2))
        Cmax_model_ = np.mean(Cmax_model)
        tmax_model_ = np.mean(tmax_model)
    except:
        Cmax_model_ = []
        tmax_model_ = []
    
    
    return Cmax_, tmax_, Cmax_model_, tmax_model_

def CdPeakSpring_(self):

    Cmax = []
    tmax = []
    try:
        tests = list(self.data.keys())
        A = self.meta['A']
        for test in tests: 
            U0  = self.meta[test]['U0']
            idx1 = np.argmin(abs(self.data[test][:,0]*self.meta[test]['tnorm_fac']))
            idx2 = np.argmin(abs(self.data[test][:,0]*self.meta[test]['tnorm_fac']-2))
            maxIdx = np.argmax(self.data[test][idx1:idx2,1]*self.meta['m']*2/(self.meta['rho']*A*U0**2))
            Cmax.append(self.meta['g']*self.data[test][maxIdx+idx1,1]*self.meta['m']*2/(self.meta['rho']*A*U0**2))
            tmax.append(self.data[test][maxIdx+idx1,0]*self.meta[test]['tnorm_fac'])
        
        Cmax_ = np.mean(Cmax)
        tmax_ = np.mean(tmax)
    except:
        pass    
    
    Cmax_model = []
    tmax_model = []
    Cmax_model2 = []
    try:
        for height in self.height: 
            U0 = np.sqrt(2*self.meta['g']*int(height)/100)
            tidx = np.argmax(self.model[height][:,1]) 
            tmax_model.append(self.meta['tSim'][tidx])
            tmax = self.model[height][tidx,0]*self.meta[test]['tnorm_fac']
            Cmax_model.append(self.meta['g']*self.model[height][tidx,-1]*self.meta['m1']*2/(self.meta['rho']*self.meta['A']*U0**2))
            Cmax_model2.append(self.meta['g']*self.model[height][tidx,2]*self.meta['m2']*2/(self.meta['rho']*self.meta['A']*U0**2))
        
        Cmax_model_ = np.mean(Cmax_model)
        Cmax_model2_ = np.mean(Cmax_model2)
        tmax_model_ = np.mean(tmax_model)
    except:
        pass
    
    return Cmax_, tmax_, Cmax_model_, tmax_model_


def GainPhase_(self):

    freq = []
    gain = []
    phase = []

    for height in self.height: 
        U0      = np.sqrt(2*self.meta['g']*int(height)/100)
        freq.append(U0/self.meta['h'])
        tidxm1  = np.argmax(self.model[height][:,2]) 
        tidxm2  = np.argmax(self.model[height][:,-1])
        gain.append(20* math.log10((self.model[height][tidxm1,2]*self.meta['m1'])/(self.model[height][tidxm2,-1]*self.meta['m2'])))
        tm1     = self.meta['tSim'][tidxm1]
        tm2     = self.meta['tSim'][tidxm2]
        phase.append(360*(1/tm2)*(tm2-tm1))

    return freq, gain, phase