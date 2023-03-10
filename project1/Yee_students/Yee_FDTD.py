import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2
from scipy.integrate import dblquad
import scipy.constants as const

class Yee_FDTD:
    
    def __init__(self, mesh, duration, Courant_factor=1, boundary_condition='PBC'):
        
        self.mesh=mesh
        self.boundary_condition=boundary_condition #PEC or PBC
        
        self.timestep=Courant_factor/np.sqrt(1/np.min(mesh.primary_x_delta)**2+1/np.min(mesh.primary_y_delta)**2)#time is scaled with a factor c!
        self.N_steps=int(duration/self.timestep)
        self.t=np.arange(self.N_steps)*self.timestep
        
        self.epsilon_r=np.ones(mesh.grid_ez[0].shape)
        self.mu_r=np.ones(mesh.grid_ez[0].shape)
        
        self.kappa_PML_y=np.ones(mesh.grid_ez[0].shape)
        self.kappa_PML_x=np.ones(mesh.grid_ez[0].shape)
        
        self.sigma_PML_y=np.zeros(mesh.grid_ez[0].shape)
        self.sigma_PML_x=np.zeros(mesh.grid_ez[0].shape)
        
        self.observation_points=[]
        self.observation_pointsij=[]
        
        self.source_point_x=0
        self.source_point_y=0
        self.source_point_i=0
        self.source_point_j=0
        self.source_amplitude=0
        self.source_value=np.zeros(self.N_steps)
        self.source_sigma=0
        
        self.output=[]
        
        
    def add_dielectric(self,epsilon_r,mu_r,x_lower,x_upper,y_lower,y_upper):
        # adds a rectangle with relative permitivity epsilon_r to the problem under consideration
        # x_lower, x_upper, y_lower, y_upper define the boundaries of the rectangle
        
        i_lower=np.argmin(np.abs(self.mesh.x_vector_hy-x_lower))
        i_upper=np.argmin(np.abs(self.mesh.x_vector_hy-x_upper))
        
        j_lower=np.argmin(np.abs(self.mesh.y_vector_hx-y_lower))
        j_upper=np.argmin(np.abs(self.mesh.y_vector_hx-y_upper))
        
        #note that effective width of the dielectric is slightly different than x_upper-x_lower
        
        self.epsilon_r[i_lower+1:i_upper+1,j_lower+1:j_upper+1]=epsilon_r
        self.mu_r[i_lower+1:i_upper+1,j_lower+1:j_upper+1]=mu_r

    def add_PML(self, N_PML, kappa_max,m=4):
        #adds a PML of N_PML layers, with kappa_max. For m and sigma_max the values from the course notes p37 are used
        
        Z_0=np.sqrt(const.mu_0/const.epsilon_0)
        
        sigma_max_x=(m+1)/150/np.pi/self.mesh.x_step*Z_0 
        sigma_max_y=(m+1)/150/np.pi/self.mesh.y_step*Z_0
        
        self.mesh.add_PML(N_PML) #expand the mesh
        
        self.epsilon_r=np.r_[np.outer(np.ones(N_PML),self.epsilon_r[0,:]),self.epsilon_r,np.outer(np.ones(N_PML),self.epsilon_r[-1,:])] #add extra rows
        self.epsilon_r=np.c_[np.ones((self.epsilon_r.shape[0],N_PML)),self.epsilon_r,np.ones((self.epsilon_r.shape[0],N_PML))] #add extra columns 
        
        self.mu_r=np.r_[np.outer(np.ones(N_PML),self.mu_r[0,:]),self.mu_r,np.outer(np.ones(N_PML),self.mu_r[-1,:])]#add extra rows
        self.mu_r=np.c_[np.ones((self.mu_r.shape[0],N_PML)),self.mu_r,np.ones((self.mu_r.shape[0],N_PML))]#add extra columns
        
        index_back=np.arange(N_PML)+1
        index_front=-np.arange(N_PML)+N_PML
        
        kappa_front=1+(kappa_max-1)*np.power((index_front/N_PML),m) #formula used from the course notes p37
        sigma_front=sigma_max_x*np.power((index_front/N_PML),m) #formula used from the course notes p37
        
        kappa_back=1+(kappa_max-1)*np.power((index_back/N_PML),m)
        sigma_back=sigma_max_x*np.power((index_back/N_PML),m)
        
        self.kappa_PML_x=np.c_[np.ones((self.kappa_PML_x.shape[0],N_PML)),self.kappa_PML_x,np.ones((self.kappa_PML_x.shape[0],N_PML))]#add extra columns
        self.sigma_PML_x=np.c_[np.zeros((self.sigma_PML_x.shape[0],N_PML)),self.sigma_PML_x,np.zeros((self.sigma_PML_x.shape[0],N_PML))]#add extra columns
        
        self.kappa_PML_x=np.r_[np.outer(kappa_front,np.ones(self.kappa_PML_x.shape[1])),self.kappa_PML_x,np.outer(kappa_back,np.ones(self.kappa_PML_x.shape[1]))] #add PML rows
        self.sigma_PML_x=np.r_[np.outer(sigma_front,np.ones(self.sigma_PML_x.shape[1])),self.sigma_PML_x,np.outer(sigma_back,np.ones(self.sigma_PML_x.shape[1]))] #add PML rows
        
        sigma_front=sigma_max_y*np.power((index_front/N_PML),m)
        sigma_back=sigma_max_y*np.power((index_back/N_PML),m)
        self.kappa_PML_y=np.r_[np.ones((N_PML,self.kappa_PML_y.shape[1])),self.kappa_PML_y,np.ones((N_PML,self.kappa_PML_y.shape[1]))]#add extra rows
        self.sigma_PML_y=np.r_[np.zeros((N_PML,self.sigma_PML_y.shape[1])),self.sigma_PML_y,np.zeros((N_PML,self.sigma_PML_y.shape[1]))]#add extra rows
        
        self.kappa_PML_y=np.c_[np.outer(np.ones(self.kappa_PML_y.shape[0]),kappa_front),self.kappa_PML_y,np.outer(np.ones(self.kappa_PML_y.shape[0]),kappa_back)] # add PML colmuns
        self.sigma_PML_y=np.c_[np.outer(np.ones(self.sigma_PML_y.shape[0]),sigma_front),self.sigma_PML_y,np.outer(np.ones(self.sigma_PML_y.shape[0]),sigma_back)] #add PML columns
        
    def add_source(self,x_source,y_source,amplitude,source_sigma,source_type='Gaussian', central_frequency=0):
        # adds a current source at position (x_source, y_source), two different source types are possible, as described in the project description
                
        self.source_point_i=np.argmin(np.abs(self.mesh.x_vector_ez-x_source))
        self.source_point_j=np.argmin(np.abs(self.mesh.y_vector_ez-y_source))
        
        self.source_point_x=self.mesh.x_vector_ez[self.source_point_i]
        self.source_point_y=self.mesh.y_vector_ez[self.source_point_j]
        
        self.source_amplitude=amplitude
        self.source_sigma=source_sigma
        self.omega_max=3/self.source_sigma
        
        
        t_c=5*self.source_sigma 
        
        if source_type=='Gaussian':
            self.source_value[:]=amplitude*np.exp(-np.power(self.t-t_c,2)/2/self.source_sigma**2)/(self.mesh.dual_x_delta[self.source_point_i]*self.mesh.dual_y_delta[self.source_point_j])

        if source_type=='Gaussian_modulated':
            self.source_value[:]=amplitude*np.exp(-np.power(self.t-t_c,2)/2/self.source_sigma**2)*np.sin(central_frequency*2*np.pi*self.t)/(self.mesh.dual_x_delta[self.source_point_i]*self.mesh.dual_y_delta[self.source_point_j])

        
    def add_observation(self,x_obs,y_obs):
        #adds an observation point at position (x_obs,y_obs)
        observation_point_i=np.argmin(np.abs(self.mesh.x_vector_ez-x_obs))
        observation_point_j=np.argmin(np.abs(self.mesh.y_vector_ez-y_obs)) 
        
        x_pos=self.mesh.x_vector_ez[observation_point_i]
        y_pos=self.mesh.y_vector_ez[observation_point_j]
        
        self.output.append(np.zeros(self.N_steps))
        self.observation_pointsij.append([observation_point_i,observation_point_j])
        self.observation_points.append([x_pos,y_pos]) 
        
    def run(self):
        # function iterates over the update equation for N_steps, 
        #the values of the electric field at the observation points 
        #are stored in self.output
        #only PBC is implemented
            
        if self.boundary_condition=='PBC':
            
            #initialization of all the fields
            
            m=self.mesh.grid_ez[0].shape[0]
            n=self.mesh.grid_ez[0].shape[1]
            Ez1new=np.zeros((m,n))
            Ez1old=np.zeros((m,n))
            Ez=np.zeros((m,n))
        
            Hx1new=np.zeros((m,n))
            Hx1old=np.zeros((m,n))
            Hx=np.zeros((m,n))
        
            Hy1new=np.zeros((m,n))
            Hy1old=np.zeros((m,n))
            Hy=np.zeros((m,n))
            
            Jz=np.zeros((m,n))
            
            #constants of update equation 1
            a1=(self.kappa_PML_x/self.timestep-self.sigma_PML_x/2)/(self.kappa_PML_x/self.timestep+self.sigma_PML_x/2)
            a2=np.reciprocal(np.outer(self.mesh.dual_x_delta,self.mesh.dual_y_delta)*self.epsilon_r*(self.kappa_PML_x/self.timestep+self.sigma_PML_x/2))
            a3=-np.reciprocal(self.epsilon_r*(self.kappa_PML_x/self.timestep+self.sigma_PML_x/2))
            
            #constants of update equation 2
            b1=np.divide(self.kappa_PML_y/self.timestep-self.sigma_PML_y/2,self.kappa_PML_y/self.timestep+self.sigma_PML_y/2)
            b2=1/self.timestep*np.reciprocal(self.kappa_PML_y/self.timestep+self.sigma_PML_y/2)
            
            #interpolations needed for update equations 3 and 4
            indexplusn=np.mod(np.arange(n)+1,n)
            
            mu_r_hx=(self.mu_r*self.mesh.dual_y_delta[np.newaxis,:]+self.mu_r[:,indexplusn]*self.mesh.dual_y_delta[np.newaxis,indexplusn])/(self.mesh.dual_y_delta[np.newaxis,:]+self.mesh.dual_y_delta[np.newaxis,indexplusn])
            kappa_PML_y_hx=(self.kappa_PML_y*self.mesh.dual_y_delta[np.newaxis,:]+self.kappa_PML_y[:,indexplusn]*self.mesh.dual_y_delta[np.newaxis,indexplusn])/(self.mesh.dual_y_delta[np.newaxis,:]+self.mesh.dual_y_delta[np.newaxis,indexplusn])
            sigma_PML_y_hx=(self.sigma_PML_y*self.mesh.dual_y_delta[np.newaxis,:]+self.sigma_PML_y[:,indexplusn]*self.mesh.dual_y_delta[np.newaxis,indexplusn])/(self.mesh.dual_y_delta[np.newaxis,:]+self.mesh.dual_y_delta[np.newaxis,indexplusn])
            kappa_PML_x_hx=(self.kappa_PML_x*self.mesh.dual_y_delta[np.newaxis,:]+self.kappa_PML_x[:,indexplusn]*self.mesh.dual_y_delta[np.newaxis,indexplusn])/(self.mesh.dual_y_delta[np.newaxis,:]+self.mesh.dual_y_delta[np.newaxis,indexplusn])
            sigma_PML_x_hx=(self.sigma_PML_x*self.mesh.dual_y_delta[np.newaxis,:]+self.sigma_PML_x[:,indexplusn]*self.mesh.dual_y_delta[np.newaxis,indexplusn])/(self.mesh.dual_y_delta[np.newaxis,:]+self.mesh.dual_y_delta[np.newaxis,indexplusn])
            
            #constants of update equation 3
            c1=-np.outer(self.mesh.dual_x_delta,np.reciprocal(self.mesh.primary_y_delta))*np.reciprocal(mu_r_hx*kappa_PML_y_hx/self.timestep+mu_r_hx*sigma_PML_y_hx/2)
            c2=np.divide(mu_r_hx*kappa_PML_y_hx/self.timestep-mu_r_hx*sigma_PML_y_hx/2,mu_r_hx*kappa_PML_y_hx/self.timestep+mu_r_hx*sigma_PML_y_hx/2)
            
            #constants of update equation 4
            d1=(kappa_PML_x_hx/self.timestep+sigma_PML_x_hx/2)*self.timestep
            d2=(-kappa_PML_x_hx/self.timestep+sigma_PML_x_hx/2)*self.timestep
            
            #interpolations needed for update equations 5 and 6
            indexplusm=np.mod(np.arange(m)+1,m)
            
            mu_r_hy=(self.mu_r*self.mesh.dual_x_delta[:,np.newaxis]+self.mu_r[indexplusm,:]*self.mesh.dual_x_delta[indexplusm,np.newaxis])/(self.mesh.dual_x_delta[:,np.newaxis]+self.mesh.dual_x_delta[indexplusm,np.newaxis])
            kappa_PML_y_hy=(self.kappa_PML_y*self.mesh.dual_x_delta[:,np.newaxis]+self.kappa_PML_y[indexplusm,:]*self.mesh.dual_x_delta[indexplusm,np.newaxis])/(self.mesh.dual_x_delta[:,np.newaxis]+self.mesh.dual_x_delta[indexplusm,np.newaxis])
            sigma_PML_y_hy=(self.sigma_PML_y*self.mesh.dual_x_delta[:,np.newaxis]+self.sigma_PML_y[indexplusm,:]*self.mesh.dual_x_delta[indexplusm,np.newaxis])/(self.mesh.dual_x_delta[:,np.newaxis]+self.mesh.dual_x_delta[indexplusm,np.newaxis])
            kappa_PML_x_hy=(self.kappa_PML_x*self.mesh.dual_x_delta[:,np.newaxis]+self.kappa_PML_x[indexplusm,:]*self.mesh.dual_x_delta[indexplusm,np.newaxis])/(self.mesh.dual_x_delta[:,np.newaxis]+self.mesh.dual_x_delta[indexplusm,np.newaxis])
            sigma_PML_x_hy=(self.sigma_PML_x*self.mesh.dual_x_delta[:,np.newaxis]+self.sigma_PML_x[indexplusm,:]*self.mesh.dual_x_delta[indexplusm,np.newaxis])/(self.mesh.dual_x_delta[:,np.newaxis]+self.mesh.dual_x_delta[indexplusm,np.newaxis])
            
            #constants of update equation 5
            e1=self.timestep*np.outer(np.reciprocal(self.mesh.primary_x_delta),self.mesh.dual_y_delta)*np.reciprocal(mu_r_hy)
            
            #constants of update equation 6
            f1=np.divide(kappa_PML_x_hy/self.timestep-sigma_PML_x_hy/2,kappa_PML_x_hy/self.timestep+sigma_PML_x_hy/2)
            f2=np.divide(kappa_PML_y_hy/self.timestep+sigma_PML_y_hy/2,kappa_PML_x_hy/self.timestep+sigma_PML_x_hy/2)
            f3=np.divide(-kappa_PML_y_hy/self.timestep+sigma_PML_y_hy/2,kappa_PML_x_hy/self.timestep+sigma_PML_x_hy/2)
            
            
            for iteratie in np.arange(self.N_steps):
                Jz[self.source_point_i,self.source_point_j]=self.source_value[iteratie]
                Ez1new[:,:]=self.Update1PBC(Ez1old,Jz,a1,a2,a3,Hy,Hx,m,n)[:,:]
                Ez[:,:]=self.Update2PBC(Ez,b1,b2,Ez1new,Ez1old,m,n)[:,:]
                Hx1new[:,:]=self.Update3PBC(Hx1old,c1,c2,Ez,m,n)[:,:]
                Hx[:,:]=self.Update4PBC(Hx,d1,d2,Hx1new,Hx1old,m,n)[:,:]
                Hy1new[:,:]=self.Update5PBC(Hy1old,e1,Ez,m,n)[:,:]
                Hy[:,:]=self.Update6PBC(Hy,f1,f2,f3,Hy1new,Hy1old,m,n)[:,:]
                
                #Changing new values of the fields to old
                Ez1old[:,:]=Ez1new[:,:]
                Hx1old[:,:]=Hx1new[:,:]
                Hy1old[:,:]=Hy1new[:,:]
                
                #Storing the values of the electric field at the observation points
                for index, observation_point in enumerate(self.observation_pointsij):
                    self.output[index][iteratie]=Ez[observation_point[0],observation_point[1]]

     
    def Update1PBC(self,Ez2old,Jz,a1,a2,a3,Hy,Hx,m,n):
        indexminm=np.mod(np.arange(m)-1,m)
        indexminn=np.mod(np.arange(n)-1,n)
        return a1*Ez2old[:,:]+a3*Jz[:,:]+a2*(Hy-Hy[indexminm,:]-Hx+Hx[:,indexminn])
    def Update2PBC(self,Ez,b1,b2,Ez1new,Ez1old,m,n):
        return b1[:,:]*Ez[:,:]+b2[:,:]*(Ez1new[:,:]-Ez1old[:,:])
    def Update3PBC(self,Hx1old,c1,c2,Ez,m,n):
        indexplusn=np.mod(np.arange(n)+1,n)
        return c2*Hx1old[:,:]+c1[:,:]*(Ez[:,indexplusn]-Ez[:,:])
    def Update4PBC(self,Hx,d1,d2,Hx1new,Hx1old,m,n):
        return Hx+d1*Hx1new+d2*Hx1old
    def Update5PBC(self,Hy1old,e1,Ez,m,n):
        indexplusm=np.mod(np.arange(m)+1,m)
        return Hy1old+e1*(Ez[indexplusm,:]-Ez)
    def Update6PBC(self,Hy,f1,f2,f3,Hy1new,Hy1old,m,n):
        return f1*Hy+f2*Hy1new+f3*Hy1old
      
        
        
        
        