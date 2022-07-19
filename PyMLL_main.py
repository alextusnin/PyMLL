import matplotlib.pyplot as plt
import numpy as np
import sys, os
from scipy.constants import pi, c, hbar, e
import ctypes
import scipy.io
from scipy.optimize import curve_fit
from scipy.linalg import dft

class Gain:
    def __init__(self):
        gain_data_mat = scipy.io.loadmat(os.path.abspath(__file__)[:-14]+'/gain_data/gain_per_m.mat')
        wavelength_mat = scipy.io.loadmat(os.path.abspath(__file__)[:-14]+'/gain_data/wavelength.mat')
        self.gain_data = gain_data_mat['gain_per_m'][0] #For Power
        self.wavelength = wavelength_mat['filtered_x'][0]*1e-9
        self.frequency = c/self.wavelength
        self.gain_rad_Hz = 0
        self.P_th=0.0008#W
    def InitZeroGain(self):
        self.gain_data = np.zeroslike(self.wavelength)
    def Transform_to_rad_Hz(self,group_index):
        self.gain_rad_Hz = c*self.gain_data/group_index
    def FitGaussian(self):
        def Gaussian(x, a1,a2, w01, w02, sigma1,sigma2):
            return  a1*np.exp(-(x-w01)**2/2/sigma1**2)#+a1*np.exp(-(x-w02)**2/2/sigma2**2)
        def Gaussian(x, a1, w01, sigma1):
            return  a1*np.exp(-(x-w01)**2/2/sigma1**2)#+a1*np.exp(-(x-w02)**2/2/sigma2**2)
        
        xdata = self.frequency
        ydata = self.gain_rad_Hz
        a1min = 100*2*np.pi*1e6
        a1max = 10*a1min
        a2min = 50*2*np.pi*1e6/1e2
        a2max = 10*a2min
        #bounds=((a1min,a2min,190e12,194e12,0.01e12,0.1e12),(a1max,a2max,193e12,195.4e12,5e12,0.5e12))
        bounds=((a1min,190e12,0.01e12),(a1max,193e12,5e12,))
        popt, pcov = curve_fit(Gaussian, xdata, ydata, bounds=bounds)
        return popt

class GLE(Gain):
    def __init__(self):
        Gain.__init__(self)
        self.n0 = 0
        self.n2 = 0
        self.FSR = 0
        self.width = 0
        self.height = 0
        self.kappa_0 = 0
        self.kappa_ex = np.array([])
        self.gain = np.array([])
        self.Dint = np.array([])
        #Auxiliary physical parameters
        self.Tr = 0
        self.Aeff = 0
        self.Leff = 0
        self.Veff = 0
        self.g0 = 0
        self.gamma = 0
        self.kappa = self.kappa_0 + self.kappa_ex
        self.N_points = len(self.Dint)
    
        self.phi = np.array([])
        
        self.D2 = 0
        self.D3 = 0
        
        self.D2_mod = 0
        
        self.n2t = 0
        self.t_th=0
        self.gain = 0
        self.frequency_grid=0

        #plt.plot(gain.frequency,gain.gain_rad_Hz/2/np.pi/1e6)
        #plt.plot(frequency_grid,(a1*np.exp(-(frequency_grid-w1)**2/2/sigma1**2)+a2*np.exp(-(frequency_grid-w2)**2/2/sigma2**2))/2/np.pi/1e6,'.')
        
    def Init_From_File(self,data_dir):
        simulation_parameters={}
        map2d=np.array([],dtype=complex)
        
        for file in os.listdir(data_dir+'class_parameters/'):
            if file.endswith('.npy'):
                if file!='gain.npy':
                    key = os.path.splitext(file)[0]
                    print(file + " is open")
                    self.__dict__[key] = np.load(data_dir+'class_parameters/'+file)
                else:
                    pass
        for file in os.listdir(data_dir+'sim_parameters/'):
            if file.endswith('.npy'):
                key = os.path.splitext(file)[0]
                simulation_parameters[key] = np.load(data_dir+'sim_parameters/'+file)
        map2d=np.load(data_dir+'map2d.npy')
        
        #self.gain = Gain()
        self.gain.Transform_to_rad_Hz(self.n0)
        a1,w1,sigma1=self.gain.FitGaussian()
        self.nu0=w1
        self.frequency_grid = self.nu0+ self.FSR*self.mu
        
        self.frequency_grid = self.nu0 + np.fft.fftshift(self.mu)*self.FSR
        
        #self.gain_grid = np.fft.fftshift((a1*np.exp(-(self.frequency_grid-w1)**2/2/sigma1**2)+a2*np.exp(-(self.frequency_grid-w2)**2/2/sigma2**2)))
        self.gain_grid = np.fft.fftshift((a1*np.exp(-(self.frequency_grid-w1)**2/2/sigma1**2)))
        
        return simulation_parameters, map2d
    def Init_From_Dict(self, resonator_parameters):
        #Physical parameters initialization
        #
        #self.gain = Gain()
        
        self.n0 = resonator_parameters['n0']
        self.n2 = resonator_parameters['n2']
        self.FSR = resonator_parameters['FSR']
        
        self.width = resonator_parameters['width']
        self.height = resonator_parameters['height']
        self.kappa_0 = resonator_parameters['kappa_0']
        self.kappa_ex = resonator_parameters['kappa_ex']
        self.Dint = np.fft.ifftshift(resonator_parameters['Dint'])
        #self.gain = np.fft.ifftshift(resonator_parameters['gain'])
        
        self.Transform_to_rad_Hz(self.n0)
     
        
        
        
        #Auxiliary physical parameters
        self.nu0 = self.frequency[np.argmax(self.gain_data)]
        self.w0 = self.nu0*2*np.pi
        self.Tr = 1/self.FSR #round trip time
        self.Aeff = self.width*self.height 
        self.Leff = c/self.n0*self.Tr 
        self.Veff = self.Aeff*self.Leff 
        self.g0 = hbar*self.w0**2*c*self.n2/self.n0**2/self.Veff
        self.gamma = self.n2*self.w0/c/self.Aeff
        self.kappa = self.kappa_0 + self.kappa_ex
        self.N_points = len(self.Dint)
        self.mu = np.int_(np.fft.fftshift(np.arange(-self.N_points/2, self.N_points/2)))
        self.phi = np.linspace(0,2*np.pi,self.N_points)
        
        
        #a1,a2,w1,w2,sigma1,sigma2=self.gain.FitGaussian()
        a1,w1,sigma1=self.FitGaussian()
        self.nu0=w1
        self.frequency_grid = self.nu0+ self.FSR*self.mu
        
        self.frequency_grid = self.nu0 + np.fft.fftshift(self.mu)*self.FSR
        
        #self.gain_grid = np.fft.fftshift((a1*np.exp(-(self.frequency_grid-w1)**2/2/sigma1**2)+a2*np.exp(-(self.frequency_grid-w2)**2/2/sigma2**2)))
        self.gain_grid = np.fft.fftshift((a1*np.exp(-(self.frequency_grid-w1)**2/2/sigma1**2)))
        
  
    def InitGainJacobian(self,A=[0]):
        
        if len(A)==1:
            A = np.zeros(self.N_points)
        Pth = self.gain.P_th#*(1./(hbar*self.w0))*(2*self.g0/self.kappa.max())
            
        
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
        
        Jacob_Gain1 = np.zeros([self.N_points,self.N_points],dtype=complex)
        Jacob_Gain2 = np.zeros([self.N_points,self.N_points],dtype=complex)
        Jacob_Gain3 = np.zeros([self.N_points,self.N_points],dtype=complex)
        Jacob_Gain4 = np.zeros([self.N_points,self.N_points],dtype=complex)
        
        
        Jacob_Gain_F1 = np.zeros([self.N_points,self.N_points],dtype=complex)
        Jacob_Gain_F2 = np.zeros([self.N_points,self.N_points],dtype=complex)
        Jacob_Gain_F3 = np.zeros([self.N_points,self.N_points],dtype=complex)
        Jacob_Gain_F4 = np.zeros([self.N_points,self.N_points],dtype=complex)
        
        Jacob = np.zeros([self.N_points*2,self.N_points*2],dtype=complex)
        
        Jacob_Gain_F1[index_1,index_1] = (self.gain_grid/2*1/(1+np.sum(np.abs(A)**2)/Pth)-self.kappa/2) -self.gain_grid/2* (2*np.real(A)**2/Pth*1/(1+np.sum(np.abs(A)**2)/Pth)**2)
        Jacob_Gain_F2[index_1,index_1]= -self.gain_grid/2* (2*np.real(A)*np.imag(A)/Pth*1/(1+np.sum(np.abs(A)**2)/Pth)**2)
        Jacob_Gain_F3[index_1,index_1] = (self.gain_grid/2*1/(1+np.sum(np.abs(A)**2)/Pth)-self.kappa/2) -self.gain_grid/2* (2*np.imag(A)**2/Pth*1/(1+np.sum(np.abs(A)**2)/Pth)**2)
        Jacob_Gain_F4[index_1,index_1]= -self.gain_grid/2* (2*np.real(A)*np.imag(A)/Pth*1/(1+np.sum(np.abs(A)**2)/Pth)**2)
        
        
        for column_ind in index_1:
            Jacob_Gain1[:,column_ind] = np.fft.fft(Jacob_Gain_F1[:,column_ind])
            Jacob_Gain2[:,column_ind] = np.fft.fft(Jacob_Gain_F2[:,column_ind])
            Jacob_Gain3[:,column_ind] = np.fft.fft(Jacob_Gain_F3[:,column_ind])
            Jacob_Gain4[:,column_ind] = np.fft.fft(Jacob_Gain_F4[:,column_ind])
        
        
        for column_ind in index_1:
            Jacob_Gain_F1[:,column_ind] = np.fft.fft(np.conj(Jacob_Gain1.T)[:,column_ind])
            Jacob_Gain_F2[:,column_ind] = np.fft.fft(np.conj(Jacob_Gain2.T)[:,column_ind])
            Jacob_Gain_F3[:,column_ind] = np.fft.fft(np.conj(Jacob_Gain3.T)[:,column_ind])
            Jacob_Gain_F4[:,column_ind] = np.fft.fft(np.conj(Jacob_Gain4.T)[:,column_ind])
        
        
        
        Jacob_Gain1 = np.conj(Jacob_Gain_F1.T)/self.N_points
        Jacob_Gain2 = np.conj(Jacob_Gain_F2.T)/self.N_points
        Jacob_Gain3 = np.conj(Jacob_Gain_F3.T)/self.N_points
        Jacob_Gain4 = np.conj(Jacob_Gain_F4.T)/self.N_points
        
        
        
        
        
        Jacob[:self.N_points,:self.N_points] = Jacob_Gain1
        Jacob[:self.N_points,self.N_points:] = Jacob_Gain2
        
        Jacob[self.N_points:,self.N_points:] =  Jacob_Gain4
        Jacob[self.N_points:,:self.N_points] = Jacob_Gain3
        
        
        return np.real(Jacob)
    def InitMaxGainJacobian(self):
        
       
        
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
        
        Jacob_Gain1 = np.zeros([self.N_points,self.N_points],dtype=complex)
       
        
        
        Jacob_Gain_F1 = np.zeros([self.N_points,self.N_points],dtype=complex)
       
        
        Jacob = np.zeros([self.N_points*2,self.N_points*2],dtype=complex)
        
        Jacob_Gain_F1[index_1,index_1] = (self.gain_grid/2) 
       
        
        
        for column_ind in index_1:
            Jacob_Gain1[:,column_ind] = np.fft.fft(Jacob_Gain_F1[:,column_ind])
            
        
        for column_ind in index_1:
            Jacob_Gain_F1[:,column_ind] = np.fft.fft(np.conj(Jacob_Gain1.T)[:,column_ind])
            
        
        
        Jacob_Gain1 = np.conj(Jacob_Gain_F1.T)/self.N_points
        
        
        
        
        
        Jacob[:self.N_points,:self.N_points] = Jacob_Gain1
        Jacob[self.N_points:,self.N_points:] =  Jacob_Gain1
        
        
        
        return np.real(Jacob)
    
    def InitDispJacobian(self):
        
        
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
        
        
        Jacob_Disp = np.zeros([self.N_points,self.N_points],dtype=complex)
        
        
        Jacob_Disp_F = np.zeros([self.N_points,self.N_points],dtype=complex)
        Jacob = np.zeros([self.N_points*2,self.N_points*2],dtype=complex)
        
        
        Jacob_Disp_F[index_1,index_1] = -self.Dint[index_1]
        
        for column_ind in index_1:
        
            Jacob_Disp[:,column_ind] = np.fft.fft(Jacob_Disp_F[:,column_ind])
        for column_ind in index_1:
        
            Jacob_Disp_F[:,column_ind] = np.fft.fft(np.conj(Jacob_Disp.T)[:,column_ind])
        
        
        Jacob_Disp = np.conj(Jacob_Disp_F.T)/self.N_points
        
        
        
        
        Jacob[:self.N_points,self.N_points:] = -Jacob_Disp
        
        Jacob[self.N_points:,:self.N_points] = -Jacob[:self.N_points,self.N_points:]
        
        return np.real(Jacob)
    def InitNonlinearJacobian(self,Psi):
        
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
        
        Jacob = np.zeros([self.N_points*2,self.N_points*2])
        Jacob[index_1,index_1] = -2*np.real(Psi)*np.imag(Psi)
        Jacob[index_1,index_2] = - np.abs(Psi)**2 - 2*np.imag(Psi)**2
        Jacob[index_2,index_2] = 2*np.real(Psi)*np.imag(Psi)
        Jacob[index_2,index_1] = np.abs(Psi)**2+2*np.real(Psi)**2
        
        return Jacob
    def NewtonRaphson(self,A_input,tol=1e-5,max_iter=50):
        A_guess = np.fft.ifft(A_input)#*np.sqrt(2*self.g0/self.kappa.max())/self.N_points
        DispJacob = (self.InitDispJacobian())*2/self.kappa.max()
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
        Aprev = np.zeros(2*self.N_points)
        Aprev[:self.N_points] = np.real(A_guess)
        Aprev[index_2] = np.imag(A_guess)
        Ak = np.zeros(Aprev.size)
        buf= np.zeros(Aprev.size)
        buf_prev= np.zeros(Aprev.size)
        diff = self.N_points
        counter =0
        diff_array=[]
        while diff>tol:
            GainJacob=self.InitGainJacobian(Aprev[index_1]+1j*Aprev[index_2])*2/self.kappa.max()
            buf = np.dot(DispJacob+GainJacob,Aprev)
            buf[index_1]-=self.g0/(hbar*self.w0)*2/self.kappa.max()*(Aprev[index_1]*Aprev[index_1]+Aprev[index_2]*Aprev[index_2])*Aprev[index_2]
            buf[index_2]+=self.g0/(hbar*self.w0)*2/self.kappa.max()*(Aprev[index_1]*Aprev[index_1]+Aprev[index_2]*Aprev[index_2])*Aprev[index_1]
            J=DispJacob+GainJacob+self.InitNonlinearJacobian(Aprev[index_1]+1j*Aprev[index_2])*self.g0/(hbar*self.w0)*2/self.kappa.max()
            Ak = Aprev - np.linalg.solve(J,buf)
            
            diff = np.sqrt(abs((Ak-Aprev).dot(np.conj(Ak-Aprev))/(Ak.dot(np.conj(Ak)))))
            print(diff)
            diff_array += [diff]
            Aprev[:] = Ak[:]
            buf_prev[:]=buf[:]
            #Aprev[index_2] = np.conj(Aprev[index_1])
            counter +=1
            
            if counter>max_iter:
                print("Did not coverge in " + str(max_iter)+ " iterations, relative error is " + str(diff))
                res = np.zeros(self.N_points,dtype=complex)
                res = Ak[index_1] + 1j*Ak[index_2]
                
                #return np.fft.fft(res)/np.sqrt(2*self.g0/self.kappa.max()), diff_array
                return np.fft.fft(res), diff_array
                break
        print("Converged in " + str(counter) + " iterations, relative error is " + str(diff))
        res = np.zeros(self.N_points,dtype=complex)
        res = Ak[index_1] + 1j*Ak[index_2]
        
        
        return np.fft.fft(res)/np.sqrt(2*self.g0/self.kappa.max()), diff_array
            
    def LinearStability(self,solution,plot_eigvals=True):
        DispJacob = (self.InitDispJacobian())*2/self.kappa.max()
        
        A_guess = np.fft.ifft(solution)*np.sqrt(2*self.g0/self.kappa.max())/self.N_points
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
        A = np.zeros(2*self.N_points)
        A[:self.N_points] = np.real(A_guess)
        A[index_2] = np.imag(A_guess)
        GainJacob=self.InitGainJacobian(A[index_1]+1j*A[index_2])*2/self.kappa.max()
        J=DispJacob+GainJacob+self.InitNonlinearJacobian(A[index_1]+1j*A[index_2])
        
        eig_vals,eig_vec = np.linalg.eig(J)
        
        eigen_vectors = np.zeros([self.N_points,2*self.N_points],dtype=complex)
        if plot_eigvals==True:
            plt.scatter(np.real(eig_vals),np.imag(eig_vals))
            plt.xlabel('Real part')
            plt.ylabel('Imaginary part')
            
        for jj in range(2*self.N_points):
            eigen_vectors[:,jj]=(eig_vec[:self.N_points,jj]).T
            eigen_vectors[:,jj]=np.fft.fft(eigen_vectors[:,jj])
        
        return eig_vals[:-1]*self.kappa.max()/2, np.fft.fft(eigen_vectors)/np.sqrt(2*self.g0/self.kappa.max())
        
        
    def Save_Data(self,map2d,Simulation_Params,directory='./'):
        params = self.__dict__
        try: 
            os.mkdir(directory+'class_parameters/')
            os.mkdir(directory+'sim_parameters/')
        except:
            pass
        for key in params.keys():
            np.save(directory+'class_parameters/'+key+'.npy',params[key])
        for key in Simulation_Params:
            np.save(directory+'sim_parameters/'+key+'.npy',Simulation_Params[key])
        np.save(directory+'map2d.npy',map2d)
        
        #print(params.keys())
    def noise(self, a):
#        return a*np.exp(1j*np.random.uniform(-1,1,self.N_points)*np.pi)
        return a*(np.random.uniform(-1,1,self.N_points) + 1j*np.random.uniform(-1,1,self.N_points))
    
    def Propagate_PseudoSpectralSAMCLIB(self, simulation_parameters, Seed=[0], dt=5e-4):
        #start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        
        eps = simulation_parameters['noise_level']
        
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        nn = simulation_parameters['Number of points']
        
        
        
        #seed = np.zeros(self.N_points,dtype=complex)
        print(np.sum(np.abs(Seed[:])))
        seed=Seed
        print(np.sum(np.abs(seed[:])))
        #seed*=np.sqrt(2*self.g0/self.kappa.max())
        seed+= self.noise(eps)#*np.sqrt(2*self.g0/self.kappa.max())
        print(np.sum(np.abs(seed[:])))
        #plt.plot(abs(seed))
        ### renormalization
        T_rn = (self.kappa.max()/2)*T
        t_st = np.float_(T_rn)/nn
        
        sol = np.ndarray(shape=(nn, self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = (seed)#/self.N_points
        print(np.sum(np.abs(sol[0,:]))**2)
        
        P_th=self.P_th#*(1./(hbar*self.w0))*(2*self.g0/self.kappa.max())
        print('Saturation power is',P_th)
        #%% crtypes defyning
        GLE_core = ctypes.CDLL(os.path.abspath(__file__)[:-14]+'/lib/lib_gle_core.so')
        GLE_core.Propagate_SAM.restype = ctypes.c_void_p
        #%% defining the ctypes variables
        
        A = np.fft.ifft(seed)#*self.N_points
        print(np.sum(np.abs(A[:]))**2)
        #plt.plot(abs(A))
        
        In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
        In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
        In_phi = np.array(self.phi,dtype=ctypes.c_double)
        In_Nphi = ctypes.c_int(self.N_points)
        In_atol = ctypes.c_double(abtol)
        In_rtol = ctypes.c_double(reltol)
        
        
        In_Ndet = ctypes.c_int(nn)
        In_Dint = np.array(self.Dint*2/self.kappa,dtype=ctypes.c_double)
        In_gain = np.array(self.gain_grid/self.kappa,dtype=ctypes.c_double)
        In_P_th = ctypes.c_double(P_th)
        In_Ttotal = ctypes.c_double(T_rn)
        In_g0 = ctypes.c_double(self.g0/(hbar*self.w0)*2/self.kappa.max())
        #In_g0 = ctypes.c_double(1.0)
        In_Nt = ctypes.c_int(int(t_st/dt)+1)
        In_dt = ctypes.c_double(dt)
        In_noise_amp = ctypes.c_double(eps)
        
        
            
        if self.n2t!=0:
            In_kappa = ctypes.c_double(self.kappa_0+self.kappa_ex)
            In_t_th = ctypes.c_double(self.t_th)
            In_n2 = ctypes.c_double(self.n2)
            In_n2t = ctypes.c_double(self.n2t)
            
            
            
        In_res_RE = np.zeros(nn*self.N_points,dtype=ctypes.c_double)
        In_res_IM = np.zeros(nn*self.N_points,dtype=ctypes.c_double)
        
        double_p=ctypes.POINTER(ctypes.c_double)
        In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
        In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
        In_phi_p = In_phi.ctypes.data_as(double_p)
        
        In_Dint_p = In_Dint.ctypes.data_as(double_p)
        In_gain_p = In_gain.ctypes.data_as(double_p)
        
        In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
        In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
        #%%running simulations
        if self.n2t==0:
            GLE_core.Propagate_SAM(In_val_RE_p, In_val_IM_p, In_phi_p, In_Dint_p, In_g0, In_gain_p, In_P_th, In_Ndet, In_Nt, In_dt,  In_Ttotal, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        else:
            #GLE_core.PropagateThermalSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_t_th, In_kappa, In_n2, In_n2t, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
            pass
        ind_modes = np.arange(self.N_points)
        for ii in range(0,nn):
            sol[ii,ind_modes] = np.fft.fft(In_res_RE[ii*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points+ind_modes])
            
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol#/np.sqrt(2*self.g0/self.kappa)
        elif out_param == 'fin_res':
            return sol[-1, :]#/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')
            
    def Propagate_PseudoSpectralStiffCLIB(self, simulation_parameters, Solver='Sie', Seed=[0], dt=5e-4):
        #start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        
        eps = simulation_parameters['noise_level']
        
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        nn = simulation_parameters['Number of points']
        
        
        
        
        
        #seed = np.zeros(self.N_points,dtype=complex)
        seed=Seed*np.sqrt(2*self.g0/self.kappa.max())
        seed+= self.noise(eps)*np.sqrt(2*self.g0/self.kappa.max())
        
        DispJacobian = self.InitDispJacobian()*2/self.kappa.max()
        GainJacobian = self.InitMaxGainJacobian()*2/self.kappa.max()
        
        DispJacobianIn = np.zeros((2*self.N_points)**2)
        GainJacobianIn = np.zeros((2*self.N_points)**2)
        index = np.arange(2*self.N_points)
        for ii in range(2*self.N_points):
            DispJacobianIn[ii*2*self.N_points + index] = DispJacobian[ii,index]
            GainJacobianIn[ii*2*self.N_points + index] = GainJacobian[ii,index]
        #plt.plot(abs(seed))
        ### renormalization
        T_rn = (self.kappa.max()/2)*T
        t_st = np.float_(T_rn)/nn
        
        sol = np.ndarray(shape=(nn, self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = (seed)/self.N_points
        print(np.sum(np.abs(sol[0,:])))
        
        #P_th=self.gain.P_th/(hbar*self.w0)
        P_th=self.gain.P_th*(1./(hbar*self.w0))*(2*self.g0/self.kappa.max())
        print('Nomralized saturation gain is ', P_th)
        #%% crtypes defyning
        GLE_core = ctypes.CDLL(os.path.abspath(__file__)[:-14]+'/lib/lib_gle_core.so')
        GLE_core.Propagate_SAM.restype = ctypes.c_void_p
        #%% defining the ctypes variables
        
        A = np.fft.ifft(seed)#*self.N_points
        
        #plt.plot(abs(A))
        
        In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
        In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
        In_phi = np.array(self.phi,dtype=ctypes.c_double)
        In_Nphi = ctypes.c_int(self.N_points)
        In_atol = ctypes.c_double(abtol)
        In_rtol = ctypes.c_double(reltol)
        In_DispJacobian = np.array(DispJacobianIn,dtype=ctypes.c_double)
        In_GainJacobian = np.array(GainJacobianIn,dtype=ctypes.c_double)
        
        In_Ndet = ctypes.c_int(nn)
        In_Dint = np.array(self.Dint*2/self.kappa,dtype=ctypes.c_double)
        In_gain = np.array(self.gain_grid/self.kappa,dtype=ctypes.c_double)
        In_P_th = ctypes.c_double(P_th)
        In_Tstep = ctypes.c_double(t_st)
        #In_g0 = ctypes.c_double(self.g0*2/self.kappa.max())
        In_g0 = ctypes.c_double(1.0)
        In_Nt = ctypes.c_int(int(t_st/dt)+1)
        In_dt = ctypes.c_double(dt)
        In_noise_amp = ctypes.c_double(eps)
        
        
            
        if self.n2t!=0:
            In_kappa = ctypes.c_double(self.kappa_0+self.kappa_ex)
            In_t_th = ctypes.c_double(self.t_th)
            In_n2 = ctypes.c_double(self.n2)
            In_n2t = ctypes.c_double(self.n2t)
            
            
            
        In_res_RE = np.zeros(nn*self.N_points,dtype=ctypes.c_double)
        In_res_IM = np.zeros(nn*self.N_points,dtype=ctypes.c_double)
        
        double_p=ctypes.POINTER(ctypes.c_double)
        In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
        In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
        In_phi_p = In_phi.ctypes.data_as(double_p)
        
        In_Dint_p = In_Dint.ctypes.data_as(double_p)
        In_gain_p = In_gain.ctypes.data_as(double_p)
        In_DispJacobian_p = In_DispJacobian.ctypes.data_as(double_p)
        In_GainJacobian_p = In_GainJacobian.ctypes.data_as(double_p)
        
        In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
        In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
        #%%running simulations
        if Solver=='Sie':
            GLE_core.Propagate_StiffSie(In_val_RE_p, In_val_IM_p, In_phi_p, In_Dint_p, In_g0, In_gain_p, In_P_th, In_DispJacobian_p, In_GainJacobian_p, In_Ndet, In_Nt, In_dt, In_Tstep, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        elif Solver=='Ross':
            GLE_core.Propagate_StiffRoss(In_val_RE_p, In_val_IM_p, In_phi_p, In_Dint_p, In_g0, In_gain_p, In_P_th, In_Jacobian_p, In_GainJacobian_p, In_Ndet, In_Nt, In_dt, In_Tstep, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
            
        
        ind_modes = np.arange(self.N_points)
        for ii in range(0,nn):
            sol[ii,ind_modes] = np.fft.fft(In_res_RE[ii*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points+ind_modes])
            
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol/np.sqrt(2*self.g0/self.kappa)
        elif out_param == 'fin_res':
            return sol[-1, :]/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')
            
class TwoCoupledGLE():
    def __init__(self,Cavity1, Cavity2,J):
        self.Cavity1 = Cavity1
        self.Cavity2 = Cavity2
        self.J = J
    def Propagate_PseudoSpectralSAMCLIB(self, simulation_parameters, Seed=[0,0], dt=5e-4):
        #start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        
        eps = simulation_parameters['noise_level']
        
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        nn = simulation_parameters['Number of points']
        
        
        
        #seed = np.zeros(self.N_points,dtype=complex)
        print(np.sum(np.abs(Seed[:])))
        seed=Seed
        print(np.sum(np.abs(seed[:])))
        #seed*=np.sqrt(2*self.g0/self.kappa.max())
        seed[:,0]+= self.Cavity1.noise(eps)#*np.sqrt(2*self.g0/self.kappa.max())
        seed[:,1]+= self.Cavity2.noise(eps)
        print(np.sum(np.abs(seed[:])))
        #plt.plot(abs(seed))
        ### renormalization
        T_rn = np.max( [self.Cavity1.kappa.max(), self.Cavity2.kappa.max()  ])/2*T#(self.kappa.max()/2)*T
        t_st = np.float_(T_rn)/nn
        
        sol = np.ndarray(shape=(nn, self.N_points,2), dtype='complex') # define an array to store the data
        sol[0,:,:] = (seed)#/self.N_points
        print(np.sum(np.abs(sol[0,:,0]))**2+np.sum(np.abs(sol[0,:,1]))**2)
        
        P_th1=self.Cavity1.P_th#*(1./(hbar*self.w0))*(2*self.g0/self.kappa.max())
        P_th2=self.Cavity2.P_th
        print('Saturation power 1 is',P_th1)
        print('Saturation power 2 is',P_th2)
        #%% crtypes defyning
        GLE_core = ctypes.CDLL(os.path.abspath(__file__)[:-14]+'/lib/lib_coupled_gle_core.so')
        GLE_core.Propagate_SAM.restype = ctypes.c_void_p
        #%% defining the ctypes variables
        
        A = np.zeros([self.Cavity1.N_points+self.Cavity2.N_points],dtype=complex)
        A[:self.Cavity1.N_points] = np.fft.ifft(seed[:,0])#*self.N_points
        A[self.Cavity1.N_points:] = np.fft.ifft(seed[:,1])#*self.N_points
        print(np.sum(np.abs(A[:]))**2)
        #plt.plot(abs(A))
        
        In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
        In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
        In_phi = np.array(self.Cavity1.phi,dtype=ctypes.c_double)
        In_Nphi = ctypes.c_int(self.Cavity1.N_points)
        In_atol = ctypes.c_double(abtol)
        In_rtol = ctypes.c_double(reltol)
        
        
        In_Ndet = ctypes.c_int(nn)
        In_Dint = np.array(2*self.Cavity1.N_points)
        In_Dint1 = self.Cavity1.Dint
        In_Dint2 = self.Cavity2.Dint
        #In_Dint = np.array(self.Dint*2/self.kappa,dtype=ctypes.c_double)
        #In_gain = np.array(self.gain_grid/self.kappa,dtype=ctypes.c_double)
        In_gain1 = self.Cavity1.gain_grid/
        In_gain2[self.Cavity1.N_points:] = Cavity2.Dint
        
        In_P_th1 = ctypes.c_double(P_th1)
        In_P_th2 = ctypes.c_double(P_th2)
        In_Ttotal = ctypes.c_double(T_rn)
        In_g0 = ctypes.c_double(self.g0/(hbar*self.w0)*2/self.kappa.max())
        #In_g0 = ctypes.c_double(1.0)
        In_Nt = ctypes.c_int(int(t_st/dt)+1)
        In_dt = ctypes.c_double(dt)
        In_noise_amp = ctypes.c_double(eps)
        
        
            
        if self.n2t!=0:
            In_kappa = ctypes.c_double(self.kappa_0+self.kappa_ex)
            In_t_th = ctypes.c_double(self.t_th)
            In_n2 = ctypes.c_double(self.n2)
            In_n2t = ctypes.c_double(self.n2t)
            
            
            
        In_res_RE = np.zeros(nn*self.N_points,dtype=ctypes.c_double)
        In_res_IM = np.zeros(nn*self.N_points,dtype=ctypes.c_double)
        
        double_p=ctypes.POINTER(ctypes.c_double)
        In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
        In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
        In_phi_p = In_phi.ctypes.data_as(double_p)
        
        In_Dint_p = In_Dint.ctypes.data_as(double_p)
        In_gain_p = In_gain.ctypes.data_as(double_p)
        
        In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
        In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
        #%%running simulations
        if self.n2t==0:
            GLE_core.Propagate_SAM(In_val_RE_p, In_val_IM_p, In_phi_p, In_Dint_p, In_g0, In_gain_p, In_P_th, In_Ndet, In_Nt, In_dt,  In_Ttotal, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        else:
            #GLE_core.PropagateThermalSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_t_th, In_kappa, In_n2, In_n2t, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
            pass
        ind_modes = np.arange(self.N_points)
        for ii in range(0,nn):
            sol[ii,ind_modes] = np.fft.fft(In_res_RE[ii*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points+ind_modes])
            
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol#/np.sqrt(2*self.g0/self.kappa)
        elif out_param == 'fin_res':
            return sol[-1, :]#/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')    
    