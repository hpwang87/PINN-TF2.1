# -*- coding: utf-8 -*-
"""
Created Jun14 2021

@author: H.P. Wang
github:  https://github.com/hpwang87
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
from pinns import NavierStokes3DPINNs
from pinns import NavierStokes2DPINNs
from funcs import flow2D





def predict_NS3D_PIV():
    # read the data
    data_pathname = './data/HIT'
    data_filename = 'HIT_data_example.mat'
    data = h5py.File(os.path.join(data_pathname, data_filename),'r')
    data_xmesh = np.transpose(data['xmesh'])
    data_ymesh = np.transpose(data['ymesh'])  
    data_zmesh = np.transpose(data['zmesh'])  
    u_data = np.transpose(data['all_data_u'])
    v_data = np.transpose(data['all_data_v'])  
    w_data = np.transpose(data['all_data_w'])
    data_num = data['datanum'][0,0]
    data_freq = data['freq'][0,0]
    dt = 1/data_freq
    dim_flag = '2d2c'
    
    # read the PINN network
    save_file = './weights/HIT_PINNs_2D2C'
    domain = sio.loadmat(save_file+'_paras.mat',squeeze_me=True)
    alpha = domain['alpha']
    norm_paras = domain['norm_paras']
    lx_min = norm_paras[0,1]
    lx_max = norm_paras[1,1]
    ly_min = norm_paras[0,2]
    ly_max = norm_paras[1,2]
    lz_min = norm_paras[0,3]
    lz_max = norm_paras[1,3]
    lt_min = norm_paras[0,0]
    lt_max = norm_paras[1,0]


    # the predicted mesh is the same as the PIV mesh
    xmesh = data_xmesh
    ymesh = data_ymesh 
    zmesh = data_zmesh 
    dns_freq = data_freq
    dt = 1/dns_freq       
    x_pred = xmesh.flatten()[:,None]
    y_pred = ymesh.flatten()[:,None]
    z_pred = zmesh.flatten()[:,None]
  
    
  
    """
    parameters
    """
    hp = {'layers':[4] + 11*[128] + [4],
          'ExistModel':1,
          'train':False,          
          'map_name':'rnn',
          'savename':'HIT_PINNs_2D2C',
          'Re':10150,
          'alpha':1,
          'dim_flag':dim_flag,             # 2d2c->no w component
          'norm_paras':norm_paras,
          'nt_lr':0.8,
          'nt_max_iternum':0,
          'nt_steps_per_loop':500,
          'nt_batch_size':10000,
          'tf_epochs':100,
          'tf_initial_epoch':0,
          'tf_steps_per_epoch':10,
          'tf_batch_size':1000,
          'tf_init_lr':1e-3}
    
    # Load trained neural network
    data = np.zeros([10,8])
    eqns = np.zeros([10,4])
    pinn_model = NavierStokes3DPINNs(hp, data, eqns)
        
        
    freq = 1/dt
    count = -1
    sizD = xmesh.shape
    tvec = dt*np.arange(0,data_num,1)
    sizT = tvec.size
    mask = np.zeros((sizD[0],sizD[1]))
   
    all_data_u = np.zeros((sizD[0],sizD[1],sizT))
    all_data_v = np.zeros((sizD[0],sizD[1],sizT))
    all_data_w = np.zeros((sizD[0],sizD[1],sizT))
    all_data_p = np.zeros((sizD[0],sizD[1],sizT))
    
    for tt in tvec:
        count = count+1
        t_pred = tt*np.ones_like(x_pred)
        pred = np.concatenate((t_pred,x_pred,y_pred,z_pred), 1)
     
        # prediction
        u_pred, v_pred, w_pred, p_pred = pinn_model.predict(pred)
        if np.mod(count,10) == 0:
            print("--- loop %d in total %d ---" % (count, sizT))

        all_data_u[:,:,count] = u_pred.reshape(sizD[0],sizD[1])  
        all_data_v[:,:,count] = v_pred.reshape(sizD[0],sizD[1])
        all_data_w[:,:,count] = w_pred.reshape(sizD[0],sizD[1])
        all_data_p[:,:,count] = p_pred.reshape(sizD[0],sizD[1])
        
    # save the predicted data
    filepath = './predict_results'
    filename = ('%s_predict.mat') % (hp['savename'])
    sio.savemat(os.path.join(filepath, filename), {'xmesh':xmesh, 'ymesh':ymesh,\
                                       'zmesh':zmesh, 'all_data_u':all_data_u,\
                                       'all_data_v':all_data_v, 'all_data_w':all_data_w,\
                                       'all_data_p':all_data_p, 'freq':freq,\
                                       'mask':mask})
        
        
    # # plot the data
    # dd = 5
    # umin = np.min(u_data)
    # umax = np.max(u_data)
    # vmin = np.min(v_data)
    # vmax = np.max(v_data) 
    # wmin = np.min(w_data)
    # wmax = np.max(w_data)
    # plt.figure()
    # ax1 = plt.subplot(2,3,1)
    # ax2 = plt.subplot(2,3,2)
    # ax3 = plt.subplot(2,3,3)
    # p1 = ax1.pcolor(data_xmesh,data_ymesh,u_data[:,:,dd],cmap='RdYlGn_r',shading='auto', vmin=umin, vmax=umax)
    # ax1.set_title(r'$u_{ori}$',fontsize=12,color='r')
    # ax1.axis('equal')
    # p2 = ax2.pcolor(data_xmesh,data_ymesh,v_data[:,:,dd],cmap='RdYlGn_r', shading='auto', vmin=vmin, vmax=vmax)
    # ax2.set_title(r'$v_{ori}$',fontsize=12,color='r')
    # ax2.axis('equal')  
    # p3 = ax3.pcolor(data_xmesh,data_ymesh,w_data[:,:,dd],cmap='RdYlGn_r', shading='auto', vmin=wmin, vmax=wmax)
    # ax3.set_title(r'$w_{ori}$',fontsize=12,color='r')
    # ax3.axis('equal') 
    # plt.colorbar(p1, ax=ax1)
    # plt.colorbar(p2, ax=ax2)
    # plt.colorbar(p3, ax=ax3)
    
    # ax4 = plt.subplot(2,3,4)
    # ax5 = plt.subplot(2,3,5) 
    # ax6 = plt.subplot(2,3,6)    
    # p4 = ax4.pcolor(xmesh[:,:,0],ymesh[:,:,0],all_data_u[:,:,dd],cmap='RdYlGn_r', shading='auto', vmin=umin, vmax=umax)
    # ax4.set_title(r'$u_{pre}$',fontsize=12,color='r')
    # ax4.axis('equal')
    # p5 = ax5.pcolor(xmesh[:,:,0],ymesh[:,:,0],all_data_v[:,:,dd],cmap='RdYlGn_r', shading='auto', vmin=vmin, vmax=vmax)
    # ax5.set_title(r'$v_{pre}$',fontsize=12,color='r')
    # ax5.axis('equal')  
    # p6 = ax6.pcolor(xmesh[:,:,0],ymesh[:,:,0],all_data_w[:,:,dd],cmap='RdYlGn_r', shading='auto', vmin=wmin, vmax=wmax)
    # ax6.set_title(r'$w_{pre}$',fontsize=12,color='r')
    # ax6.axis('equal')      
    # plt.colorbar(p4, ax=ax4)
    # plt.colorbar(p5, ax=ax5)
    # plt.colorbar(p6, ax=ax6)   
    # plt.show()
    
    # plt.savefig('./predict_results/NS3DPIV_compare.png', dpi=300)
    # plt.show()
        
        
        




def predict_Hemi3D_PIV():
    # read the PIV mesh
    data_pathname = './data/hemisphere'
    data_filename = 'hemisphere_3D3CTOMO_example.mat'
    data = h5py.File(os.path.join(data_pathname, data_filename),'r')
    xmesh = np.transpose(data['dns_xmesh'])
    ymesh = np.transpose(data['dns_ymesh'])  
    zmesh = np.transpose(data['dns_zmesh'])  
    mask = np.transpose(data['mask']) 
    data_num = data['datanum'][0,0]
    data_freq = data['freq'][0,0]
    dt = 1/data_freq
    x_pred = xmesh.flatten()[:,None]
    y_pred = ymesh.flatten()[:,None]
    z_pred = zmesh.flatten()[:,None]
    
    # read the PINN network
    save_file = './weights/Hemisphere_3D3CTOMO'
    domain = sio.loadmat(save_file+'_paras.mat',squeeze_me=True)
    alpha = domain['alpha']
    norm_paras = domain['norm_paras']

    """
    parameters
    """
    hp = {'layers':[4] + 15*[128] + [4],
          'ExistModel':1,
          'train':False,          
          'map_name':'rnn',
          'savename':'Hemisphere_3D3CTOMO',
          'Re':2750,
          'alpha':1,
          'dim_flag':'3d3c',             # 2d2c->no w component
          'norm_paras':norm_paras,
          'nt_lr':0.8,
          'nt_max_iternum':0,
          'nt_steps_per_loop':20,
          'nt_batch_size':10000,
          'tf_epochs':100,
          'tf_initial_epoch':0,
          'tf_steps_per_epoch':10,
          'tf_batch_size':5000,
          'tf_init_lr':1e-3}
    
    # Load trained neural network
    data = np.zeros([10,8])
    eqns = np.zeros([10,4])
    pinn_model = NavierStokes3DPINNs(hp, data, eqns)
        
        
    freq = 1/dt
    count = -1
    sizD = xmesh.shape
    tvec = dt*np.arange(0,data_num,1)
    sizT = tvec.size
    # 三维数据
    all_data_u = np.zeros((sizD[0],sizD[1],sizD[2],sizT),dtype=np.float32)
    all_data_v = np.zeros((sizD[0],sizD[1],sizD[2],sizT),dtype=np.float32)
    all_data_w = np.zeros((sizD[0],sizD[1],sizD[2],sizT),dtype=np.float32)
    all_data_p = np.zeros((sizD[0],sizD[1],sizD[2],sizT),dtype=np.float32)
    
    for tt in tvec:
        count = count+1
        t_pred = tt*np.ones_like(x_pred)
        pred = np.concatenate((t_pred,x_pred,y_pred,z_pred), 1)
     
        # prediction
        u_pred, v_pred, w_pred, p_pred = pinn_model.predict(pred)
        if np.mod(count,10) == 0:
            print("--- loop %d in total %d ---" % (count, sizT))

        tmp = u_pred.reshape(sizD[0],sizD[1],sizD[2])
        tmp[mask==1] = np.nan
        all_data_u[:,:,:,count] = tmp  
        tmp = v_pred.reshape(sizD[0],sizD[1],sizD[2])
        tmp[mask==1] = np.nan
        all_data_v[:,:,:,count] = tmp  
        tmp = w_pred.reshape(sizD[0],sizD[1],sizD[2])
        tmp[mask==1] = np.nan
        all_data_w[:,:,:,count] = tmp  
        tmp = p_pred.reshape(sizD[0],sizD[1],sizD[2])
        tmp[mask==1] = np.nan
        all_data_p[:,:,:,count] = tmp  

        
    # save the predicted data
    filepath = './predict_results'
    filename = ('%s_predict_alpha_%d.mat') % (hp['savename'], alpha)
    sio.savemat(os.path.join(filepath, filename), {'xmesh':xmesh, 'ymesh':ymesh,\
                                       'zmesh':zmesh, 'all_data_u':all_data_u,\
                                       'all_data_v':all_data_v, 'all_data_w':all_data_w,\
                                       'all_data_p':all_data_p, 'freq':freq,\
                                       'mask':mask})
    
    
    



def predict_Flow2D():
    lmin = 0
    lmax = 2.0*np.pi
    lnum = 64
    tmin = 0
    tmax = 1
    tnum = 64
    xvec = np.linspace(lmin, lmax, lnum)
    yvec = np.linspace(lmin, lmax, lnum)
    tvec = np.linspace(tmin, tmax, tnum) 
    xmesh,ymesh,tmesh = np.meshgrid(xvec,\
                                    yvec,\
                                    tvec,indexing='xy') 
    uvp_ture = flow2D(tmesh, xmesh, ymesh)   
    u_data = uvp_ture['u']
    v_data = uvp_ture['v']
    p_data = uvp_ture['p']
    sizD = xmesh.shape
    mask = np.zeros((sizD[0],sizD[1]))   
    freq = 1.0/(tvec[1]-tvec[0])
    xmesh,ymesh = np.meshgrid(xvec,yvec, indexing='xy')
    
    # save the theoretical solution
    all_data_u = u_data
    all_data_v = v_data
    all_data_p = p_data
    filepath = './predict_results'
    filename = 'Flow2D_exact.mat'
    sio.savemat(os.path.join(filepath, filename), {'xmesh':xmesh, 'ymesh':ymesh,
                                               'all_data_u':all_data_u,\
                                               'all_data_v':all_data_v,\
                                               'all_data_p':all_data_p, 'freq':freq,\
                                               'mask':mask})    

    nlevels = [0, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15, 0.17, 0.2]
    nlevels = [0.1]
    alphas = [1, 10, 20, 30, 40, 50, 60, 70 ,80]
    alphas = [1]

    N_data = [128, 256, 512, 1024, 2048, 4096, 8192, 8192*2, 8192*4,8192*8,8192*16] 
    N_data = [2048]
    # number of neurons
    N_cell = 64 
    for nlevel in nlevels:
        for alpha in alphas:
            for dnum in N_data:
                savename = ("Flow2DPINNs_dataN%d_cellN%d_adaptswish") % (dnum, N_cell) 
                save_file = './weights/'+savename
                domain = sio.loadmat(save_file+'_paras.mat',squeeze_me=True)
                #alpha = domain['alpha']
                norm_paras = domain['norm_paras']  
            
                """
                parameters
                """       
                hp = {'layers':[3] + 7*[N_cell] + [3],
                       'ExistModel':1,
                       'train':False,    
                       'map_name':'rnn',
                       'savename':savename,
                       'Re':1,
                       'alpha':alpha,                      
                       'norm_paras':norm_paras,
                       'nt_lr':0.8,
                       'nt_max_iternum':0,
                       'nt_steps_per_loop':500,
                       'nt_batch_size':10000,
                       'tf_epochs':10000,
                       'tf_initial_epoch':0,
                       'tf_steps_per_epoch':3,
                       'tf_batch_size':5000,
                       'tf_init_lr':1e-3}
            
                # Load trained neural network
                data = np.zeros([10,6])
                eqns = np.zeros([10,3])
                pinn_model = NavierStokes2DPINNs(hp, data, eqns)
                
                
                count = -1 
                x_pred = xmesh.flatten()[:,None]
                y_pred = ymesh.flatten()[:,None]
                # 二维数据
                all_data_u = np.zeros((sizD[0],sizD[1],sizD[2]))
                all_data_v = np.zeros((sizD[0],sizD[1],sizD[2]))
                all_data_p = np.zeros((sizD[0],sizD[1],sizD[2]))
            
                for tt in tvec:
                    count = count+1
                    t_pred = tt*np.ones_like(x_pred)
                    pred = np.concatenate((t_pred,x_pred,y_pred), 1)
             
                    # prediction
                    u_pred, v_pred, p_pred = pinn_model.predict(pred)
                    if np.mod(count,10) == 0:
                        print("--- loop %d in total %d ---" % (count, sizD[2]))
            
                    all_data_u[:,:,count] = u_pred.reshape(sizD[0],sizD[1])  
                    all_data_v[:,:,count] = v_pred.reshape(sizD[0],sizD[1])
                    all_data_p[:,:,count] = p_pred.reshape(sizD[0],sizD[1])
                
                    # save the predicted data
                    filename = ('%s_predict.mat') % (hp['savename'])
                    sio.savemat(os.path.join(filepath, filename), {'xmesh':xmesh, 'ymesh':ymesh,
                                                       'all_data_u':all_data_u,\
                                                       'all_data_v':all_data_v,\
                                                       'all_data_p':all_data_p, 'freq':freq,\
                                                       'mask':mask})
                        
                        
                # # plot the data
                # dd = 5
                # umin = np.min(u_data)
                # umax = np.max(u_data)
                # vmin = np.min(v_data)
                # vmax = np.max(v_data)    
                # pmin = np.min(p_data)
                # pmax = np.max(p_data)  
                # plt.figure()
                # ax1 = plt.subplot(2,3,1)
                # ax2 = plt.subplot(2,3,2)
                # ax3 = plt.subplot(2,3,3)
                # p1 = ax1.pcolor(xmesh,ymesh,u_data[:,:,dd],cmap='RdYlGn_r', 
                #                 shading='auto', vmin=umin, vmax=umax)
                # ax1.set_title(r'$u_{ori}$',fontsize=12,color='r')
                # ax1.axis('equal')
                # p2 = ax2.pcolor(xmesh,ymesh,v_data[:,:,dd],cmap='RdYlGn_r', 
                #                 shading='auto', vmin=vmin, vmax=vmax)
                # ax2.set_title(r'$v_{ori}$',fontsize=12,color='r')
                # ax2.axis('equal') 
                # p3 = ax3.pcolor(xmesh,ymesh,p_data[:,:,dd],cmap='RdYlGn_r', 
                #                 shading='auto', vmin=pmin, vmax=pmax)
                # ax3.set_title(r'$p_{ori}$',fontsize=12,color='r')
                # ax3.axis('equal')         
          
                # plt.colorbar(p1, ax=ax1)
                # plt.colorbar(p2, ax=ax2)
                # plt.colorbar(p3, ax=ax3)
                
                # ax4 = plt.subplot(2,3,4)
                # ax5 = plt.subplot(2,3,5)
                # ax6 = plt.subplot(2,3,6)
                # p4 = ax4.pcolor(xmesh,ymesh,all_data_u[:,:,dd],cmap='RdYlGn_r', 
                #                 shading='auto', vmin=umin, vmax=umax)
                # ax4.set_title(r'$u_{pre}$',fontsize=12,color='r')
                # ax4.axis('equal')
                # p5 = ax5.pcolor(xmesh,ymesh,all_data_v[:,:,dd],cmap='RdYlGn_r', 
                #                 shading='auto', vmin=vmin, vmax=vmax)
                # ax5.set_title(r'$v_{pre}$',fontsize=12,color='r')
                # ax5.axis('equal')  
                # tmp = all_data_p[:,:,dd]
                # p6 = ax6.pcolor(xmesh,ymesh,tmp-np.mean(tmp),cmap='RdYlGn_r', 
                #                 shading='auto', vmin=pmin, vmax=pmax)
                # ax6.set_title(r'$p_{pre}$',fontsize=12,color='r')
                # ax6.axis('equal')         
                # plt.colorbar(p4, ax=ax4)
                # plt.colorbar(p5, ax=ax5)
                # plt.colorbar(p6, ax=ax6)       
                
                
                # plt.show()
                
                # plt.savefig('./predict_results/Flow2D_compare.png', dpi=300)
                # plt.show()



if __name__ == "__main__":
    # NS3DPIV prediction
    # predict_NS3D_PIV()
    
    # NS3DPIV prediction
    # predict_Hemi3D_PIV()    
    
    # Flow2D prediction
    predict_Flow2D()


