# -*- coding: utf-8 -*-
"""
Created Jun14 2021

@author: H.P. Wang
github:  https://github.com/hpwang87
"""

import numpy as np
import random
import h5py
import os
from funcs import flow2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



# random.seed(1234)
# np.random.seed(1234)
    

def PIV_DataGenerator(data_pathname, data_filename):
    """
    read the PIV data
    """
    # Please use HDF reader for matlab v7.3 files
    # data = sio.loadmat(os.path.join(data_pathname, data_filename))
    data = h5py.File(os.path.join(data_pathname, data_filename),'r')
    piv_xmesh = np.transpose(data['xmesh'])
    piv_ymesh = np.transpose(data['ymesh'])
    piv_zmesh = np.transpose(data['zmesh'])
    piv_u = np.transpose(data['all_data_u'])
    piv_v = np.transpose(data['all_data_v'])
    piv_w = np.transpose(data['all_data_w'])
    eqn_region = np.transpose(data['dns_region'])
    piv_freq = data['freq'][0,0]
    piv_dt = 1/piv_freq
    dt = piv_dt
    piv_num = data['datanum'][0,0]
    lt_min = 0.0    
    # read the string
    dim_flag = data['dataflag'][:].tobytes()[::2].decode()
    # check the string
    if dim_flag.lower() == '2d2c' or dim_flag.lower() == '2d3c':
        """
        generate the PIV mesh: 2D2C, 2D3C 
        """
        piv_zpos = piv_zmesh[0,0]
        lt_max = dt*(piv_num-1)
        x_data,y_data,z_data,t_data = np.meshgrid(piv_xmesh[0,:],\
                                                         piv_ymesh[:,0],\
                                                         piv_zpos,\
                                                         dt*np.arange(0,piv_num,1),indexing='xy')                 
    elif  dim_flag.lower() == '3d3ctomo':  
        """
        generate the PIV mesh: 3D3C TOMO
        """
        lt_max = dt*(piv_num-1)
        x_data,y_data,z_data,t_data = np.meshgrid(piv_xmesh[0,:,0],\
                                                         piv_ymesh[:,0,0],\
                                                         piv_zmesh[0,0,:],\
                                                         dt*np.arange(0,piv_num,1),indexing='xy')   
    elif dim_flag.lower() == '3d3cptv':
        piv_tmesh = np.transpose(data['tmesh']) 
        x_data = piv_xmesh
        y_data = piv_ymesh
        z_data = piv_zmesh
        t_data = piv_tmesh
        
    x_data = x_data.flatten()[:,None]
    y_data = y_data.flatten()[:,None]
    z_data = z_data.flatten()[:,None]
    t_data = t_data.flatten()[:,None]
    u_data = piv_u.flatten()[:,None]
    v_data = piv_v.flatten()[:,None]
    w_data = piv_w.flatten()[:,None]
    # add noise
    # nlevel = 0.05
    # sizU = u_data.shape
    # stdu = np.std(u_data,axis=None,ddof=0)
    # Noise = nlevel*stdu*np.random.randn(sizU[0],sizU[1])
    # u_data = u_data+Noise
    # stdv = np.std(v_data,axis=None,ddof=0)
    # Noise = nlevel*stdv*np.random.randn(sizU[0],sizU[1])
    # v_data = v_data+Noise
    # stdw = np.std(w_data,axis=None,ddof=0)
    # Noise = nlevel*stdw*np.random.randn(sizU[0],sizU[1])
    # w_data = w_data+Noise
           
    p_data = np.zeros_like(u_data)
    N_data = 1000000
    if x_data.shape[0]<=N_data:
        data = np.concatenate((t_data, x_data, y_data, z_data, u_data, v_data, w_data, p_data), 1) 
    else:
        idx = np.random.choice(x_data.shape[0], N_data, replace=True)
        data = np.concatenate((t_data[idx,:], x_data[idx,:], y_data[idx,:], z_data[idx,:], u_data[idx,:], v_data[idx,:], w_data[idx,:], p_data[idx,:]), 1)
    del x_data, y_data, z_data, t_data
    del u_data, v_data, w_data, p_data
    
    
    """
    boundary conditions
    """
    lx_min = eqn_region[0,0]
    lx_max = eqn_region[0,1]
    ly_min = eqn_region[1,0]
    ly_max = eqn_region[1,1]
    lz_min = eqn_region[2,0]
    lz_max = eqn_region[2,1] 

    num_x = 100
    dx = (lx_max-lx_min)/(num_x-1)
    xvec = lx_min+dx*np.arange(0,num_x,1)
    xvec = xvec.flatten()[:,None]
    dz = dx
    num_z = np.floor((lz_max-lz_min)/dz)
    zvec = lz_min+dz*np.arange(0,num_z,1)  
    zvec = zvec.flatten()[:,None]
    # boundary points: y = 0
    # Retau=550
    yvec = 0.0
    # time space is the same as the PIV
    dt = piv_dt      
    tvec = lt_min+dt*np.arange(0,piv_num,1)     
    tvec = tvec.flatten()[:,None]         
    # randomly generate data points
    N_bc = 200000
    xidx = np.random.choice(xvec.shape[0], N_bc, replace=True)  
    zidx = np.random.choice(zvec.shape[0], N_bc, replace=True)
    tidx = np.random.choice(tvec.shape[0], N_bc, replace=True)
    # generate the boundary points
    x_bc = xvec[xidx,:]
    y_bc = yvec*np.ones_like(x_bc)  
    z_bc = zvec[zidx,:]
    t_bc = tvec[tidx,:]
    u_bc = np.zeros_like(x_bc)
    v_bc = np.zeros_like(x_bc)    
    w_bc = np.zeros_like(x_bc) 
    p_bc = np.zeros_like(x_bc)    
    # 计算w分量
    flag_data = np.ones_like(u_bc)
    bc = np.concatenate((t_bc, x_bc, y_bc, z_bc, u_bc, v_bc, w_bc, p_bc, flag_data), 1) 
    del x_bc, y_bc, z_bc, t_bc
    del u_bc, v_bc, w_bc, p_bc, flag_data
    
    
    """
    generate the equation points
    """
    lx_min = eqn_region[0,0]
    lx_max = eqn_region[0,1]
    ly_min = eqn_region[1,0]
    ly_max = eqn_region[1,1]
    lz_min = eqn_region[2,0]
    lz_max = eqn_region[2,1] 
   
    num_x = 1000
    dx = (lx_max-lx_min)/(num_x-1)
    xvec = lx_min+dx*np.arange(0,num_x,1)
    xvec = xvec.flatten()[:,None]
    dz = dx
    num_z = np.floor((lz_max-lz_min)/dz)
    zvec = lz_min+dz*np.arange(0,num_z+1,1)  
    zvec = zvec.flatten()[:,None]
    num_y = 1000
    dy = (ly_max-ly_min)/(num_y-1)
    yvec = ly_min+dy*np.arange(0,num_y,1)
    yvec = yvec.flatten()[:,None]    
    
    # # Retau=550,法向256
    # yloc = np.cos(np.arange(0,257,1)*np.pi/256.0)
    # yloc = 1-yloc
    # yvec = yloc[0:128]
    # yvec = yvec.flatten()[:,None]
    
    dt = piv_dt      
    tvec = lt_min+dt*np.arange(0,piv_num,1)     
    tvec = tvec.flatten()[:,None]         
    # random points for saving memory
    N_eqns = 1000000
    xidx = np.random.choice(xvec.shape[0], N_eqns, replace=True)
    yidx = np.random.choice(yvec.shape[0], N_eqns, replace=True)
    zidx = np.random.choice(zvec.shape[0], N_eqns, replace=True)
    tidx = np.random.choice(tvec.shape[0], N_eqns, replace=True)  
  
    eqns = np.concatenate((tvec[tidx,:], xvec[xidx,:], yvec[yidx,:], zvec[zidx,:]), 1)  
    del xidx, yidx, zidx, tidx
    
    
    # [min(t),min(x),min(y),min(z),mean(u),mean(v),mean(w),mean(p)]
    # [max(t),max(x),max(y),max(z),std(u),std(v),std(w),std(p)]
    norm_paras = np.zeros([2,8])
    norm_paras[0,0:4] = eqns.min(0)
    norm_paras[1,0:4] = eqns.max(0)
    norm_paras[0,4:8] = np.mean(data[:,4:8], 0)
    norm_paras[1,4:8] = np.std(data[:,4:8], 0)
    # for 2D PIV
    norm_paras[0,6] = norm_paras[0,5]
    norm_paras[1,6] = norm_paras[1,5]
    # for pressure
    norm_paras[0,7] = 0
    norm_paras[1,7] = 1   
    
    return data, eqns, bc, dim_flag, norm_paras






def Hemisphere_DataGenerator(data_pathname, data_filename):
    """
    read the hemisphere data
    """
    # Please use HDF reader for matlab v7.3 files
    # data = sio.loadmat(os.path.join(data_pathname, data_filename))
    hemi_data = h5py.File(os.path.join(data_pathname, data_filename),'r')
    piv_xmesh = np.transpose(hemi_data['xmesh'])
    piv_ymesh = np.transpose(hemi_data['ymesh'])
    piv_zmesh = np.transpose(hemi_data['zmesh'])
    piv_tmesh = np.transpose(hemi_data['tmesh'])
    piv_u = np.transpose(hemi_data['all_data_u'])
    piv_v = np.transpose(hemi_data['all_data_v'])
    piv_w = np.transpose(hemi_data['all_data_w'])
    piv_freq = hemi_data['freq'][0,0]
    piv_dt = 1/piv_freq
    dt = piv_dt
    piv_num = hemi_data['datanum'][0,0]
    """
    note：
    dim_falg='2d2c', the w component will not be considered
    dim_falg='3d3c', the w component will be considered
    """
    dim_flag = '3d3c'
        
    x_data = piv_xmesh.flatten()[:,None]
    y_data = piv_ymesh.flatten()[:,None]
    z_data = piv_zmesh.flatten()[:,None]
    t_data = piv_tmesh.flatten()[:,None]
    u_data = piv_u.flatten()[:,None]
    v_data = piv_v.flatten()[:,None]
    w_data = piv_w.flatten()[:,None]
    
    p_data = np.zeros_like(u_data)
    N_data = 10000000
    if x_data.shape[0]<=N_data:
        data = np.concatenate((t_data, x_data, y_data, z_data, u_data, v_data, w_data, p_data), 1) 
    else:
        idx = np.random.choice(x_data.shape[0], N_data, replace=True)
        data = np.concatenate((t_data[idx,:], x_data[idx,:], y_data[idx,:], z_data[idx,:], u_data[idx,:], v_data[idx,:], w_data[idx,:], p_data[idx,:]), 1)

    del x_data, y_data, z_data, t_data
    del u_data, v_data, w_data, p_data
    
    
    """
    boundary points
    """
    x_bc = np.transpose(hemi_data['bc_x'])
    y_bc = np.transpose(hemi_data['bc_y'])
    z_bc = np.transpose(hemi_data['bc_z'])
    t_bc = np.transpose(hemi_data['bc_t'])
    u_bc = np.zeros_like(x_bc)
    v_bc = np.zeros_like(x_bc)    
    w_bc = np.zeros_like(x_bc) 
    p_bc = np.zeros_like(x_bc)    

    bc = np.concatenate((t_bc, x_bc, y_bc, z_bc, u_bc, v_bc, w_bc, p_bc), 1) 
    del x_bc, y_bc, z_bc, t_bc
    del u_bc, v_bc, w_bc, p_bc
    
    
    """
    equations points
    """
    x_eqn = np.transpose(hemi_data['eqn_x'])
    y_eqn = np.transpose(hemi_data['eqn_y'])
    z_eqn = np.transpose(hemi_data['eqn_z'])
    t_eqn = np.transpose(hemi_data['eqn_t'])   
    eqns = np.concatenate((t_eqn, x_eqn, y_eqn, z_eqn), 1)  
    del x_eqn, y_eqn, z_eqn, t_eqn
    
    # [min(t),min(x),min(y),min(z),mean(u),mean(v),mean(w),mean(p)]
    # [max(t),max(x),max(y),max(z),std(u),std(v),std(w),std(p)]
    norm_paras = np.zeros([2,8])
    norm_paras[0,0:4] = eqns.min(0)
    norm_paras[1,0:4] = eqns.max(0)
    norm_paras[0,4:8] = np.mean(data[:,4:8], 0)
    norm_paras[1,4:8] = np.std(data[:,4:8], 0)
    # for 2D PIV, 
    # norm_paras[0,6] = norm_paras[0,5]
    # norm_paras[1,6] = norm_paras[1,5]
    # for pressure
    norm_paras[0,7] = 0
    norm_paras[1,7] = 1   
    
    return data, eqns, bc, dim_flag, norm_paras







def Flow2D_DataGenerator(lmin=0, lmax=np.pi, tmin=0, tmax=1.0, nlevel=0.0):
    """
    flow2D: generate data
    """
    xvec = np.linspace(lmin, lmax, 100)
    yvec = np.linspace(lmin, lmax, 100)
    tvec = np.linspace(tmin, tmax, 100)
    
    xmesh,ymesh,tmesh = np.meshgrid(xvec,\
                                    yvec,\
                                    tvec,indexing='xy') 
    
    uvp_ture = flow2D(tmesh, xmesh, ymesh)
    
    u = uvp_ture['u']
    v = uvp_ture['v']
    p = uvp_ture['p']
 
    # plt.figure()
    # ax1 = plt.subplot(1,2,1)
    # ax2 = plt.subplot(1,2,2)
    # p1 = ax1.pcolor(xmesh[:,:,0],ymesh[:,:,0],v[:,:,32],cmap='RdYlGn_r', 
    #                 shading='auto',vmin=-0.1, vmax=0.1)
    # #ax1.quiver(xmesh[:,:,0],ymesh[:,:,0], u[:,:,10], v[:,:,10])
    # ax1.set_title('orginal u',fontsize=12,color='r')
    # ax1.axis('square')
    # add noise
    sizU = u.shape
    for ii in np.arange(0,sizU[2],1):
        curu = u[:,:,ii]
        stdu = np.std(curu,axis=None,ddof=0)
        Noise = nlevel*stdu*np.random.randn(sizU[0],sizU[1])
        u[:,:,ii] = curu+Noise
        curv = v[:,:,ii]
        stdv = np.std(curv,axis=None,ddof=0)
        Noise = nlevel*stdv*np.random.randn(sizU[0],sizU[1])
        v[:,:,ii] = curv+Noise
        
    # p2 = ax2.pcolor(xmesh[:,:,0],ymesh[:,:,0],v[:,:,32],cmap='RdYlGn_r', 
    #                 shading='auto', vmin=-0.1, vmax=0.1)
    # #ax2.quiver(xmesh[:,:,0],ymesh[:,:,0], u[:,:,10], v[:,:,10])
    # ax2.set_title('noised u',fontsize=12,color='r')
    # ax2.axis('square')    
    # plt.colorbar(p1, ax=ax1)
    # plt.colorbar(p2, ax=ax2)       
    
    
    # 生成数据点:[t,x,y,u,v,p]
    x_data = xmesh.flatten()[:,None]
    y_data = ymesh.flatten()[:,None]
    t_data = tmesh.flatten()[:,None]
    u_data = u.flatten()[:,None]
    v_data = v.flatten()[:,None]
    p_data = p.flatten()[:,None]
    
    data = np.concatenate((t_data, x_data, y_data, u_data, v_data, p_data), 1)
    del x_data, y_data, t_data
    del u_data, v_data, p_data     
    
    # [min(t),min(x),min(y),mean(u),mean(v),mean(p)]
    # [max(t),max(x),max(y),std(u),std(v),std(p)]
    norm_paras = np.zeros([2,6])
    norm_paras[0,0:3] = np.array([tmin, lmin, lmin])
    norm_paras[1,0:3] = np.array([tmax, lmax, lmax])
    norm_paras[0,3:6] = np.mean(data[:,3:6], 0)
    norm_paras[1,3:6] = np.std(data[:,3:6], 0)
    # for pressure
    norm_paras[0,5] = 0
    norm_paras[1,5] = 1   
    
    return data, norm_paras





def Suboff2D_DataGenerator(data_pathname):
    """
    Suboff 2D
    """
    # Please use HDF reader for matlab v7.3 files
    # data = sio.loadmat(os.path.join(data_pathname, data_filename))
    data_filename = 'suboff_data_points_2d.mat'
    tmp = h5py.File(os.path.join(data_pathname, data_filename),'r')
    data = np.transpose(tmp['piv_data'])
    N_data = 1000000
    idx = np.random.choice(data.shape[0], N_data, replace=True)  
    data = data[idx,:]
    
    """
    read SUBOFF bc data and equations points
    """
    data_filename = 'suboff_model_points_2d.mat'
    tmp = h5py.File(os.path.join(data_pathname, data_filename),'r')
    eqns = np.transpose(tmp['eqns_points'])
    # N_eqns = 20000000
    # idx = np.random.choice(eqns.shape[0], N_eqns, replace=True)  
    # eqns = eqns[idx,:]
    
    bc_dirichlet = np.transpose(tmp['BC_dirichlet'])
    # N_dirichlet = 3000000
    # idx = np.random.choice(bc_dirichlet.shape[0], N_dirichlet, replace=True)  
    # bc_dirichlet = bc_dirichlet[idx,:]
    
    bc_neumann_x = np.transpose(tmp['BC_neumann_x'])
    # N_neumann = 1000000
    # idx = np.random.choice(bc_neumann_x.shape[0], N_neumann, replace=True)  
    # bc_neumann_x = bc_neumann_x[idx,:]    
        
    # 数据归一化参数
    # [min(t),min(x),min(y),mean(u),mean(v),mean(p)]
    # [max(t),max(x),max(y),std(u),std(v),std(p)]
    norm_paras = np.zeros([2,6])    
    norm_paras[0,0:3] = eqns.min(0)
    norm_paras[1,0:3] = eqns.max(0)
    norm_paras[0,3:6] = np.mean(data[:,3:6], 0)
    norm_paras[1,3:6] = np.std(data[:,3:6], 0)
    # 压力保持不变
    norm_paras[0,5] = 0
    norm_paras[1,5] = 1   
    
    return data, eqns, bc_dirichlet, bc_neumann_x, norm_paras



"""
for testing
"""
if __name__ == "__main__":
    # lmin = 0
    # lmax = 2.0*np.pi
    # tmin = 0
    # tmax = 2
    # data, norm_paras = Flow2D_DataGenerator(lmin=lmin, lmax=lmax, tmin=0, tmax=tmax)
    
    str = './data/Suboff/2D'
    data, eqns, bc, norm_paras = Suboff2D_DataGenerator(str)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(bc[0:-1:100,0],bc[0:-1:100,1],bc[0:-1:100,2])
    #ax.quiver(bc[0:-1:100,0],bc[0:-1:100,1],bc[0:-1:100,2],bc[0:-1:100,5],bc[0:-1:100,3],bc[0:-1:100,4])
    plt.show()
    
    # data_pathname = './data/DNS'
    # data_filename = 'data_Retau550_T500_3D3CTOMO.mat'
    # data, eqns, bc, norm_paras = PIV_DataGenerator(data_pathname, data_filename)
    