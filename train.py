# -*- coding: utf-8 -*-
"""
Created Jun14 2021

@author: H.P. Wang
github:  https://github.com/hpwang87
"""

from datagenerator import PIV_DataGenerator
from datagenerator import Flow2D_DataGenerator
from datagenerator import Suboff2D_DataGenerator
from datagenerator import Hemisphere_DataGenerator
from pinns import NavierStokes3DPINNs
from pinns import NavierStokes2DPINNs
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
from numpy.matlib import repmat
import time



def train_NS3D_PIV():
    """
    read the PIV data into the 3D PINN
    """
    data_pathname = './data/HIT'
    data_filename = 'HIT_data_example.mat'
    # the boundary conditions bc is not used
    data, eqns, bc, dim_flag, norm_paras = PIV_DataGenerator(data_pathname, data_filename)
  
    
    """
    parameters
    """
    hp = {'layers':[4] + 11*[128] + [4],
          'ExistModel':0,
          'train':True,
          'map_name':'rnn',
          'savename':'HIT_PINNs_2D2C',
          'Re':1,
          'alpha':1,
          'dim_flag':dim_flag,             # 2d2c->no w component
          'norm_paras':norm_paras,
          'nt_lr':0.8,
          'nt_max_iternum':0,
          'nt_steps_per_loop':20,
          'nt_batch_size':10000,
          'tf_epochs':100,
          'tf_initial_epoch':0,
          'tf_steps_per_epoch':10,
          'tf_batch_size':1000,
          'tf_init_lr':1e-3}
    
    t_start = time.process_time()
    pinn_model = NavierStokes3DPINNs(hp, data, eqns)

    # training and saving
    H = pinn_model.train()   
    t_end = time.process_time()
    print('Running time: %s seconds' % (t_end-t_start)) 

    # plot
    plt.figure()
    plt.axes(yscale="log")
    plt.ylim(1e-6, 1e2)
    N = np.arange(0,len(H.history['loss']))
    plt.plot(N,H.history['loss'],label='total_loss')
    #plt.scatter(N,H.history['loss'])
    plt.plot(N,H.history['loss_fn_data'],label='data_loss')
    #plt.scatter(N,H.history['val_loss'])
    plt.plot(N,H.history['loss_fn_eqns'],label='eqns_loss')   
    plt.title('Training Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.grid(linestyle='-.')
    plt.legend(loc=1)
    
    save_file = './weights/'+pinn_model.savename
    plt.savefig(save_file+'_loss.png', dpi=300)
    plt.show()
    

    


def train_Hemi3D_PIV():
    """
    read the hemispher data into PINN
    """
    data_pathname = './data/hemisphere'
    data_filename = 'hemisphere_3D3CTOMO_example.mat'
    data, eqns, bc, dim_flag, norm_paras = Hemisphere_DataGenerator(data_pathname, data_filename)
  
    
    """
    parameters
    """
    hp = {'layers':[4] + 15*[128] + [4],
          'ExistModel':0,
          'train':True,
          'map_name':'rnn',
          'savename':'Hemisphere_3D3CTOMO',
          'Re':2750,
          'alpha':1,
          'dim_flag':dim_flag,             # 2d2c->no w component
          'norm_paras':norm_paras,
          'nt_lr':0.8,
          'nt_max_iternum':0,
          'nt_steps_per_loop':20,
          'nt_batch_size':10000,
          'tf_epochs':100,
          'tf_initial_epoch':0,
          'tf_steps_per_epoch':10,
          'tf_batch_size':1000,
          'tf_init_lr':1e-3}
    
    t_start = time.process_time()
    # intial condition is None
    pinn_model = NavierStokes3DPINNs(hp, data, eqns, None, bc)

    # training and saving
    H = pinn_model.train()  
    t_end = time.process_time()
    print('Running time: %s seconds' % (t_end-t_start)) 

    # plot
    plt.figure()
    plt.axes(yscale="log")
    plt.ylim(1e-6, 1e2)
    N = np.arange(0,len(H.history['loss']))
    plt.plot(N,H.history['loss'],label='total_loss')
    #plt.scatter(N,H.history['loss'])
    plt.plot(N,H.history['loss_fn_data'],label='data_loss')
    #plt.scatter(N,H.history['val_loss'])
    plt.plot(N,H.history['loss_fn_eqns'],label='eqns_loss')   
    plt.plot(N,H.history['loss_fn_conds'],label='bc_loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.grid(linestyle='-.')
    plt.legend(loc=1)
    
    save_file = './weights/'+pinn_model.savename
    plt.savefig(save_file+'_loss.png', dpi=300)
    plt.show()
    
    
    
    
def train_NSFlow2D():
    """
    Flow2D
    """
    lmin = 0
    lmax = 2.0*np.pi
    tmin = 0
    tmax = 1

    nlevels = [0, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15, 0.17, 0.2]
    nlevels = [0]
    alphas = [1, 10, 20, 30, 40, 50, 60, 70 ,80]
    alphas = [1]
    for nlevel in nlevels:
        for alpha in alphas:
   
            data, norm_paras = Flow2D_DataGenerator(lmin=lmin, lmax=lmax, 
                         tmin=tmin, tmax=tmax, nlevel=nlevel) 
            N_eqns = 10000
            N_data = 2048
            # equation points
            xidx = np.random.choice(data.shape[0], N_eqns, replace=False)
            eqns = data[xidx,0:3]
            # data points
            xidx = np.random.choice(data.shape[0], N_data, replace=False)
            curdata = data[xidx, 0:6]     

            """
            构parameters
            """
            N_cell = 64     
            savename = ("Flow2DPINNs_dataN%d_cellN%d_adaptswish") % (N_data, N_cell)                
            
            hp = {'layers':[3] + 7*[N_cell] + [3],
                  'ExistModel':0,
                  'train':True,
                  'map_name':'rnn',
                  'savename':savename,
                  'Re':1,
                  'alpha':alpha,
                  'norm_paras':norm_paras,
                  'nt_lr':0.8,
                  'nt_max_iternum':100,
                  'nt_steps_per_loop':100,
                  'nt_batch_size':0,
                  'tf_epochs':100,
                  'tf_initial_epoch':1,
                  'tf_steps_per_epoch':3,
                  'tf_batch_size':5000,
                  'tf_init_lr':1e-3}
            
            t_start = time.process_time()
            pinn_model = NavierStokes2DPINNs(hp, curdata, eqns)
        
            # training and saving
            H = pinn_model.train()  
            t_end = time.process_time()
            print('Running time: %s seconds' % (t_end-t_start))   




def train_Suboff2D():
    """
    PINN for suboff
    """
    data_pathname = './data/Suboff/2D'
    data, eqns, bc_dirichlet, bc_neumann_x, norm_paras = Suboff2D_DataGenerator(data_pathname)
    
    data_filename = 'suboff_model_points_2d.mat'
    tmp = h5py.File(os.path.join(data_pathname, data_filename),'r')
    pred = np.transpose(tmp['pred_points'])
    x_pred = pred[:,0:1]
    y_pred = pred[:,1:2]
    tmax  = tmp['tmax'][0,0] 
    dt = tmp['dt'][0,0]
    t_pred = 0.95*tmax*np.ones_like(x_pred)
    pred = np.concatenate((t_pred,x_pred,y_pred), 1)
  
    
    """
    parameters
    """
    hp = {'layers':[3] + 11*[128] + [3],
          'ExistModel':1,
          'train':True,
          'map_name':'rnn',
          'savename':'Suboff2DPINNs',
          'Re':4.762e4,
          'alpha':1,
          'norm_paras':norm_paras,
          'nt_lr':0.8,
          'nt_max_iternum':0,
          'nt_steps_per_loop':500,
          'nt_batch_size':10000,
          'tf_epochs':10000,
          'tf_initial_epoch':0,
          'tf_steps_per_epoch':10,
          'tf_batch_size':5000,
          'tf_init_lr':1e-3}
    
    
    iternum = 1
    bc_init = bc_dirichlet
    for ii in np.arange(0,iternum,1):
        if ii == 0:
            hp['ExistModel'] = 1
            #bc_init = np.concatenate((-dt*np.ones_like(x_pred),x_pred,y_pred,1.0*np.ones_like(x_pred),0.0*np.ones_like(x_pred),0.0*np.ones_like(x_pred)), 1)
        else:
            hp['ExistModel'] = 1
        hp['tf_initial_epoch'] = 10000*ii
        hp['tf_epochs'] = 10000*(ii+1)
        pinn_model = NavierStokes2DPINNs(hp, None, eqns, bc_init, bc_dirichlet, bc_neumann_x)

        # training and saving
        H = pinn_model.train() 
        
        # predict the last fields
        u_pred, v_pred, p_pred = pinn_model.predict(pred)
        
        # generate new initial condition at -dt
        bc_init = np.concatenate((-dt*np.ones_like(x_pred),x_pred,y_pred,u_pred,v_pred,p_pred), 1)
        # bc_init = repmat(bc_init,2,1)

    # plot
    plt.figure()
    plt.axes(yscale="log")
    plt.ylim(1e-6, 1e2)
    N = np.arange(0,len(H.history['loss']))
    plt.plot(N,H.history['loss'],label='total_loss')
    #plt.scatter(N,H.history['loss'])
    plt.plot(N,H.history['loss_fn_data'],label='data_loss')
    #plt.scatter(N,H.history['val_loss'])
    plt.plot(N,H.history['loss_fn_eqns'],label='eqns_loss')   
    plt.plot(N,H.history['loss_fn_bc'],label='bc_loss')   
    plt.title('Training Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.grid(linestyle='-.')
    plt.legend(loc=1)
    
    save_file = './weights/'+pinn_model.savename
    plt.savefig(save_file+'_loss.png', dpi=300)
    plt.show()
    
    
    
    

if __name__ == "__main__":
    
    # training the PIV data
    # train_NS3D_PIV()
    
    # training the Hemisphere data
    # train_Hemi3D_PIV()
    
    # training the Flow2D data
    train_NSFlow2D()







