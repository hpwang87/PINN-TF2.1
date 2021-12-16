# -*- coding: utf-8 -*-
"""
Created Jun14 2021

@author: H.P. Wang
github:  https://github.com/hpwang87
"""

import numpy as np
from userbackend import tf, _GPU_NUM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras import backend
from maps import generator
from custom_lbfgs import lbfgs, Struct
import scipy.io as sio
from autograd_minimize.tf_wrapper import tf_function_factory
from autograd_minimize import minimize 



"""
user defined informantion print
"""
class LossPrintingCallback(Callback):
    def __init__(self):
        super(LossPrintingCallback, self).__init__()
    """
    user defined loss print function
    """
    def on_epoch_begin(self, epoch, logs=None):
        """
        do not output anything
        """
    
    def on_epoch_end(self, epoch, logs=None):
        lr = float(backend.get_value(self.model.optimizer.lr))
        print("Epoch %05d: loss: %.4e, loss_data: %.4e, loss_eqns: %.4e, loss_conds: %.4e, learning rate: %8.6f" %
                (epoch, logs["loss"], logs["loss_fn_data"], logs["loss_fn_eqns"], logs["loss_fn_conds"], lr)
        )
        
        
        

class NavierStokes3DPINNs(object):
    def __init__(self, hp, data, eqns, *conds):
        """
        data points:    [t,x,y,z,u,v,w,p]
        eqns points:    [t,x,y,z,0,0,0,0]
        conds:          other conditions like boundaries
            # ii=0, initial BC,             [t,x,y,z,u,v,w,p]
            # ii=1, Dirichlet BC,           [t,x,y,z,u,v,w,p]
            # ii=2, Neumann BC of ux,       [t,x,y,z,ux]
            # ii=3, Neumann BC of uy,       [t,x,y,z,uy]
            # ii=4, Neumann BC of uz,       [t,x,y,z,uz]            
            # ii=2, Neumann BC of vx,       [t,x,y,z,vx]
            # ii=3, Neumann BC of vy,       [t,x,y,z,vy]
            # ii=4, Neumann BC of vz,       [t,x,y,z,vz]    
            # ii=2, Neumann BC of wx,       [t,x,y,z,wx]
            # ii=3, Neumann BC of wy,       [t,x,y,z,wy]
            # ii=4, Neumann BC of wz,       [t,x,y,z,wz]    
            # ii=2, Neumann BC of px,       [t,x,y,z,px]
            # ii=3, Neumann BC of py,       [t,x,y,z,py]
            # ii=4, Neumann BC of pz,       [t,x,y,z,pz]    
            
        Note: equation points should include the points of boundary conditions
        """
        # clear session
        tf.keras.backend.clear_session()
        # hp is the structure of hyper-parameters
        self.dtype = 'float32'
        self.layers = hp['layers']
        self.ExistModel = hp['ExistModel']
        self.map_name = hp['map_name']
        self.savename = hp['savename']
        self.Re = hp['Re']
        self.alpha = hp['alpha']   
        self.dim_flag = hp['dim_flag']
        self.training = hp['train']
        self.iternum = 0

                
        # Setting up the optimizers with the hyper-parameters
        self.nt_config = Struct()
        self.nt_config.learningRate = hp["nt_lr"]
        self.nt_config.maxIter = hp["nt_max_iternum"]
        self.nt_config.stepIter = hp["nt_steps_per_loop"]
        self.nt_config.batchSize = hp["nt_batch_size"]
        self.nt_config.tolFun = 1.0 * np.finfo(float).eps
        
        
        self.tf_config = Struct()
        self.tf_config.epochs = hp['tf_epochs']
        self.tf_config.initial_epoch = hp['tf_initial_epoch']
        self.tf_config.steps_per_epoch = hp['tf_steps_per_epoch']
        self.tf_config.batch_size = hp['tf_batch_size']
        self.tf_config.init_lr = hp['tf_init_lr']
        
        # get the other conditions (boundary)
        self.conds = self.get_conditions(*conds)
        
        # merge the inputs to generate training data
        self.train_X, self.train_Y = self.merge_inputs(data, eqns)   
        
        # Initialize the loss recording list
        self.loss_all = []
        self.val_loss = []
        self.loss_data = []
        self.loss_eqns = []
        self.loss_conds = []
        self.alpha_seq = []
        # alpha_seq is initialized by alpha
        self.alpha_seq.append(self.alpha)


        # Multi GPU or single GPU
        if _GPU_NUM > 1:
            strategy = tf.distribute.MirroredStrategy()
        elif _GPU_NUM == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            
        self.tf_config.gpus_number = strategy.num_replicas_in_sync
        self.tf_config.global_batch_size = self.tf_config.batch_size*self.tf_config.gpus_number
        # learning rate
        reduce_lr = LearningRateScheduler(self.exponential_staircase_scheduler,verbose=0)
        # callback
        self.call_back_list =[
                ModelCheckpoint(filepath='./weights/', 
                                monitor='loss', save_best_only=True, period=100),
                reduce_lr,
                LossPrintingCallback()]
                #EarlyStopping(monitor='val_loss', min_delta = 0.05, patience=100)]
        # optimizer       
        self.optimizer = Adam(lr=self.tf_config.init_lr)  
        
        # multi-GPU
        if self.training:
            with strategy.scope():        
                if self.ExistModel == 0:
                    self.norm_paras = hp['norm_paras']
                    self.model = self.build_model()
                elif self.ExistModel == 1:
                    self.norm_paras = self.loadParas()  
                    self.model = self.loadNN()
               
                self.model.compile(optimizer=self.optimizer,\
                                   loss=self.loss_fn_all,\
                                   metrics=[self.loss_fn_data, self.loss_fn_eqns, self.loss_fn_conds])
                self.model.summary()
        # single-GPU      
        else:
            if self.ExistModel == 0:
                self.norm_paras = hp['norm_paras']
                self.model = self.build_model()
            elif self.ExistModel == 1:
                self.norm_paras = self.loadParas()  
                self.model = self.loadNN()
            self.model.summary()   
        

    
    
    def build_model(self):
        model = generator(self.layers, self.norm_paras, map_name=self.map_name)
     
        return model
  
    

    
    def get_conditions(self, *conds):
        """
        get the other (boundary) conditions from the inputs
        """
        conds_keys = ['init', 'dirichlet', 
                      'ux', 'uy', 'uz',
                      'vx', 'vy', 'vz',
                      'wx', 'wy', 'wz',
                      'px', 'py', 'pz']
        
        # default is no conditions
        conds_dict = dict.fromkeys(conds_keys, None)
    
        conds_num = len(conds)
        if conds_num > 0:
            # there are conditions
            for ii in np.arange(0,conds_num,1):
                # ii= 0, initial BC,             [t,x,y,z,u,v,w,p]
                # ii= 1, Dirichlet BC,           [t,x,y,z,u,v,w,p]
                # ii= 2, Neumann BC of ux,       [t,x,y,z,ux]
                # ii= 3, Neumann BC of uy,       [t,x,y,z,uy]
                # ii= 4, Neumann BC of uz,       [t,x,y,z,uz]            
                # ii= 5, Neumann BC of vx,       [t,x,y,z,vx]
                # ii= 6, Neumann BC of vy,       [t,x,y,z,vy]
                # ii= 7, Neumann BC of vz,       [t,x,y,z,vz]    
                # ii= 8, Neumann BC of wx,       [t,x,y,z,wx]
                # ii= 9, Neumann BC of wy,       [t,x,y,z,wy]
                # ii=10, Neumann BC of wz,       [t,x,y,z,wz]    
                # ii=11, Neumann BC of px,       [t,x,y,z,px]
                # ii=12, Neumann BC of py,       [t,x,y,z,py]
                # ii=13, Neumann BC of pz,       [t,x,y,z,pz]  
                if conds[ii] is None:
                    conds_dict[conds_keys[ii]] = None
                else:
                    conds_dict[conds_keys[ii]] = conds[ii]
                
        return conds_dict
            
            
    
    def merge_inputs(self, data, eqns):
        """
        data points:    [t,x,y,z,u,v,w,p]
        eqns points:    [t,x,y,z]
        
        train_X:        [t,x,y,z]
        train_Y:        [t,x,y,z,u,v,w,p,flag0,flag1]
        flag0:          flag of data points
        flag1:          flalg of equations points
        
        Note: equation points include the points of boundary conditions
        """
        if data is None: 
            # there is only equation points, data points
            eqns_num = eqns.shape[0]
            # [t, x, y, u, v, p, flag0,flag1]
            if self.dtype=='float32':
                train_Y = np.zeros([eqns_num,10], dtype=np.float32)
            elif self.dtype=='float64':
                train_Y = np.zeros([eqns_num,10], dtype=np.float64) 
            train_Y[0:eqns_num,0:4] = eqns
            train_Y[0:eqns_num,9:10] = 1.0
        else:
            # there are both eqns points and data points
            data_num = data.shape[0]
            if self.dtype=='float32':
                train_Y = np.zeros([data_num,10], dtype=np.float32)
            elif self.dtype=='float64':
                train_Y = np.zeros([data_num,10], dtype=np.float64)             
            train_Y[0:data_num,0:8] = data
            train_Y[0:data_num,8:9] = 1.0
            # adding eqns points
            eqns_num = eqns.shape[0]
            # [t, x, y, u, v, p, flag0,flag1]
            if self.dtype=='float32':
                tmp = np.zeros([eqns_num,10], dtype=np.float32)
            elif self.dtype=='float64':
                tmp = np.zeros([eqns_num,10], dtype=np.float64) 
            tmp[0:eqns_num,0:4] = eqns
            train_Y = np.concatenate([train_Y, tmp], axis=0) 
            # all the points need to estimate the residual of equations
            train_Y[:,9:10] = 1.0

        train_X = train_Y[:,0:4]
        return train_X, train_Y
    
    
  
    def ns_eqns(self, X):
        """
        Returns
        -------
        residual of Navier-Stokes equations

        """    
        t = tf.convert_to_tensor(X[:,0:1], self.dtype)
        x = tf.convert_to_tensor(X[:,1:2], self.dtype)
        y = tf.convert_to_tensor(X[:,2:3], self.dtype)
        z = tf.convert_to_tensor(X[:,3:4], self.dtype)
        
        # Using the new GradientTape paradigm of TF2.0
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t)
            tape1.watch(x)
            tape1.watch(y)
            tape1.watch(z)
                
            with tf.GradientTape(persistent=True) as tape2:
                # Watching gradients of t,x,y,z
                tape2.watch(t)
                tape2.watch(x)
                tape2.watch(y)
                tape2.watch(z)
                # Packing together the inputs
                X = tf.stack([t[:,0],x[:,0],y[:,0],z[:,0]], axis=1)
                # Getting the prediction
                # Y = self.model(X, training=self.training)
                Y = self.model(X)
                u = Y[:,0:1]
                v = Y[:,1:2]
                w = Y[:,2:3]
                p = Y[:,3:4]
                # recover to real value
                u = u*self.norm_paras[1,4]+self.norm_paras[0,4]
                v = v*self.norm_paras[1,5]+self.norm_paras[0,5]
                w = w*self.norm_paras[1,6]+self.norm_paras[0,6]
                p = p*self.norm_paras[1,7]+self.norm_paras[0,7]
            # first-order deriavative
            u_t = tape2.gradient(u, t)
            u_x = tape2.gradient(u, x)
            u_y = tape2.gradient(u, y)
            u_z = tape2.gradient(u, z)
            v_t = tape2.gradient(v, t)
            v_x = tape2.gradient(v, x)
            v_y = tape2.gradient(v, y)
            v_z = tape2.gradient(v, z)
            w_t = tape2.gradient(w, t)
            w_x = tape2.gradient(w, x)
            w_y = tape2.gradient(w, y)
            w_z = tape2.gradient(w, z)
            p_x = tape2.gradient(p, x)
            p_y = tape2.gradient(p, y)
            p_z = tape2.gradient(p, z)  
        # first-order deriavative
        u_xx = tape1.gradient(u_x, x)
        u_yy = tape1.gradient(u_y, y)
        u_zz = tape1.gradient(u_z, z)
        v_xx = tape1.gradient(v_x, x)
        v_yy = tape1.gradient(v_y, y)
        v_zz = tape1.gradient(v_z, z)
        w_xx = tape1.gradient(w_x, x)
        w_yy = tape1.gradient(w_y, y)
        w_zz = tape1.gradient(w_z, z)
            
        # delete the tape
        del tape1, tape2
        
        e1 = u_t + (u * u_x + v * u_y + w * u_z) + p_x - (1.0 / self.Re) * (u_xx + u_yy + u_zz)
        e2 = v_t + (u * v_x + v * v_y + w * v_z) + p_y - (1.0 / self.Re) * (v_xx + v_yy + v_zz)
        e3 = w_t + (u * w_x + v * w_y + w * w_z) + p_z - (1.0 / self.Re) * (w_xx + w_yy + w_zz)
        e4 = u_x + v_y + w_z
    
        # Buidling the PINNs
        return e1, e2, e3, e4           
        



    def get_gradient(self, X, flag):
        """
        Returns
        -------
        the gradient of u,v,p

        """    
        t = tf.convert_to_tensor(X[:,0:1], self.dtype)
        x = tf.convert_to_tensor(X[:,1:2], self.dtype)
        y = tf.convert_to_tensor(X[:,2:3], self.dtype)
        z = tf.convert_to_tensor(X[:,3:4], self.dtype)
        
        # Using the new GradientTape paradigm of TF2.0
        # persistent: multi-times gradint               
        with tf.GradientTape(persistent=True) as tape:
            # Watching gradients of t,x,y,z
            tape.watch(t)
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            # Packing together the inputs
            X = tf.stack([t[:,0],x[:,0],y[:,0],z[:,0]], axis=1)
            # Getting the prediction
            # Y = self.model(X, training=self.training)
            Y = self.model(X)
            u = Y[:,0:1]
            v = Y[:,1:2]
            w = Y[:,2:3]
            p = Y[:,3:4]
            # recover to real value
            u = u*self.norm_paras[1,4]+self.norm_paras[0,4]
            v = v*self.norm_paras[1,5]+self.norm_paras[0,5]
            w = w*self.norm_paras[1,6]+self.norm_paras[0,6]
            p = p*self.norm_paras[1,7]+self.norm_paras[0,7]
        # first-order deriavative
        if flag.lower() == 'ux':
            g = tape.gradient(u, x)
        elif flag.lower() == 'uy':      
            g = tape.gradient(u, y)
        elif flag.lower() == 'uz':      
            g = tape.gradient(u, z)            
        elif flag.lower() == 'vx':
            g = tape.gradient(v, x)
        elif flag.lower() == 'vy':
            g = tape.gradient(v, y)  
        elif flag.lower() == 'vz':
            g = tape.gradient(v, z)     
        elif flag.lower() == 'wx':
            g = tape.gradient(w, x)
        elif flag.lower() == 'wy':
            g = tape.gradient(w, y)  
        elif flag.lower() == 'wz':
            g = tape.gradient(w, z)              
        elif flag.lower() == 'px':
            g = tape.gradient(p, x)    
        elif flag.lower() == 'py':
            g = tape.gradient(p, y)  
        elif flag.lower() == 'pz':
            g = tape.gradient(p, z)                
    
        # delete tape
        del tape
        
        # Buidling the PINNs
        return g 
    
    
    
    def get_uvwp(self, X):
        """
        get the output of the network
        The predict function is designed for performance in large scale inputs. 
        For small amount of inputs that fit in one batch, directly using
        __call__() is recommended for faster execution, e.g., model(x), 
        or model(x, training=False)
        
        return the velocity and pressure

        """ 
        Xi = tf.convert_to_tensor(X[:,0:4], self.dtype)
        Y = self.model(Xi)
        u = Y[:,0:1]*self.norm_paras[1,4]+self.norm_paras[0,4]
        v = Y[:,1:2]*self.norm_paras[1,5]+self.norm_paras[0,5]
        w = Y[:,2:3]*self.norm_paras[1,6]+self.norm_paras[0,6]
        p = Y[:,3:4]*self.norm_paras[1,7]+self.norm_paras[0,7]
        
        return u, v, w, p
    
    
    
    def loss_fn_all(self, Y_true, Y_pred):
        """
        自定义loss
        Y_true: [t, x, y, z, u, v, w, p, flag0, flag1]
        """
        # update the iteration number
        self.iternum = self.iternum+1 
        
        # loss of equations
        le = self.loss_fn_eqns(Y_true, Y_pred)
        # loss of data
        ld = self.loss_fn_data(Y_true, Y_pred)    
        # loss of boundary condtions
        lb = self.loss_fn_conds(Y_true, Y_pred)
        # total loss                          
        return self.alpha*le+ld+lb    
 
    
 
    def loss_fn_eqns(self, Y_true, Y_pred):
        """
        return the loss of eqns
        """   
        #flag for equations
        flag_data = Y_true[:,9:10]
        num_data = tf.reduce_sum(flag_data)+1.0
        
        # loss of equations
        X = Y_true[:,0:4]
        e1,e2,e3,e4 = self.ns_eqns(X)  
        loss_eqns_e1 = tf.reduce_sum(tf.square(e1*flag_data))/num_data
        loss_eqns_e2 = tf.reduce_sum(tf.square(e2*flag_data))/num_data
        loss_eqns_e3 = tf.reduce_sum(tf.square(e3*flag_data))/num_data
        loss_eqns_e4 = tf.reduce_sum(tf.square(e4*flag_data))/num_data        
        loss_eqns = loss_eqns_e1+loss_eqns_e2+loss_eqns_e3+loss_eqns_e4

        return loss_eqns
    
    


    def loss_fn_data(self, Y_true, Y_pred):
        """
        return the loss of data
        """       
        # loss of data
        flag_data = Y_true[:,8:9]
        num_data = tf.reduce_sum(flag_data)+1.0
        
        # the data value
        ut = Y_true[:,4:5]
        vt = Y_true[:,5:6]
        wt = Y_true[:,6:7]
        # pt = Y_true[:,7:8]
        # predicted value, recover to real value
        up = Y_pred[:,0:1]*self.norm_paras[1,4]+self.norm_paras[0,4]
        vp = Y_pred[:,1:2]*self.norm_paras[1,5]+self.norm_paras[0,5]
        wp = Y_pred[:,2:3]*self.norm_paras[1,6]+self.norm_paras[0,6]
        # pp = Y_pred[:,3:4]*self.norm_paras[1,7]+self.norm_paras[0,7]
        
        tmp = tf.square(ut-up)
        loss_data_u = tf.reduce_sum(tmp*flag_data)/num_data
        tmp = tf.square(vt-vp)
        loss_data_v = tf.reduce_sum(tmp*flag_data)/num_data
        tmp = tf.square(wt-wp)
        loss_data_w = tf.reduce_sum(tmp*flag_data)/num_data        
        
        if self.dim_flag.lower() == '2d2c':
            # 2D2C PIV, no w component
            loss_data = loss_data_u+loss_data_v  
        else:
            # consider the w component
            loss_data = loss_data_u+loss_data_v+loss_data_w  
        # loss_data = loss_data_u+loss_data_v+loss_data_w  
            
        return loss_data
    
    
    
    
    def loss_fn_conds(self, Y_true, Y_pred):
        """
        return the loss of all the BCs
        # ii= 0, initial BC,             [t,x,y,z,u,v,w,p]
        # ii= 1, Dirichlet BC,           [t,x,y,z,u,v,w,p]
        # ii= 2, Neumann BC of ux,       [t,x,y,z,ux]
        # ii= 3, Neumann BC of uy,       [t,x,y,z,uy]
        # ii= 4, Neumann BC of uz,       [t,x,y,z,uz]            
        # ii= 5, Neumann BC of vx,       [t,x,y,z,vx]
        # ii= 6, Neumann BC of vy,       [t,x,y,z,vy]
        # ii= 7, Neumann BC of vz,       [t,x,y,z,vz]    
        # ii= 8, Neumann BC of wx,       [t,x,y,z,wx]
        # ii= 9, Neumann BC of wy,       [t,x,y,z,wy]
        # ii=10, Neumann BC of wz,       [t,x,y,z,wz]    
        # ii=11, Neumann BC of px,       [t,x,y,z,px]
        # ii=12, Neumann BC of py,       [t,x,y,z,py]
        # ii=13, Neumann BC of pz,       [t,x,y,z,pz]  
        """ 
        batch_size = 1000
        loss = 0.0
        # iterate to estimate the loss of Bcs
        for key, val in self.conds.items():
            if (key == 'init') or (key == 'dirichlet'):
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    idx = np.random.choice(val.shape[0], batch_size, replace=True)
                    tmpX = val[idx,0:4]
                    # using the fluctuation to calculate the error
                    ut = val[idx,4:5]
                    vt = val[idx,5:6]
                    wt = val[idx,6:7]
                    # pt = val[idx,7:8]
                    up, vp, wp, pp = self.get_uvwp(tmpX)
                    
                    tmpu = tf.reduce_mean(tf.square(ut-up))
                    tmpv = tf.reduce_mean(tf.square(vt-vp))
                    tmpw = tf.reduce_mean(tf.square(wt-wp))                   
                    loss = loss + tmpu + tmpv + tmpw
            # for the conditions of gradients: ux, uy, uz, vx, vy, vz, px, py, pz
            else:
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    idx = np.random.choice(val.shape[0], batch_size, replace=True)
                    tmpX = val[idx,0:4]
                    # using the fluctuation to calculate the error
                    gt = val[idx,4:5]
                    gp = self.get_gradient(tmpX, key)
                    
                    tmp = tf.reduce_mean(tf.square(gt-gp))
                    
                    loss = loss + tmp      
                    
        return loss
        
    
    
    
    def lbfgs_callback(self, Xi):
        # Xi is for code running

        # iteration number
        self.iternum = self.iternum+1
        
        # prediction
        X = self.cur_train_Y[:,0:4]
        Y_pred = self.model.predict(X,batch_size=8192,
                               workers=4, use_multiprocessing=True)
        
        # loss of equations
        le = self.loss_fn_eqns(self.cur_train_Y, Y_pred)
        # loss of the data
        ld = self.loss_fn_data(self.cur_train_Y, Y_pred)    
        # loss of other conditions (BCs)
        lb = self.loss_fn_conds(self.cur_train_Y, Y_pred)
        # total loss                          
        loss =  self.alpha*le+ld+lb
    
        self.loss_data.append(ld)
        self.loss_eqns.append(le)  
        self.loss_conds.append(lb)
        self.loss.append(loss)
        if self.iternum % 10 == 0:        
            print('L-BGFS-B Iter=%05d: Loss: %.4e, loss_data: %.4e, loss_eqns: %.4e, loss_conds: %.4e' %
                      (self.iternum, loss, ld, le, lb))                
      
        


    def train(self):
        self.training = True        
        history = self.model.fit(self.train_X, self.train_Y, batch_size=self.tf_config.global_batch_size, 
                                 epochs=self.tf_config.epochs, 
                                 verbose=0, 
                                 callbacks=self.call_back_list,
                                 validation_split=0.0,
                                 shuffle=True,
                                 initial_epoch=self.tf_config.initial_epoch, 
                                 steps_per_epoch=self.tf_config.steps_per_epoch,
                                 workers=self.tf_config.gpus_number, use_multiprocessing=True)
        
        self.loss_data = history.history['loss_fn_data']
        self.loss_eqns = history.history['loss_fn_eqns']
        self.loss_conds = history.history['loss_fn_conds']
        self.loss = history.history['loss']
        # self.val_loss = history.history['val_loss']
        
        # save model
        self.saveNN()  
        # save the parameters
        sio.savemat('./weights/'+self.savename+'_paras.mat', 
                    {'batch_size':self.tf_config.batch_size,
                     'epochs':self.tf_config.epochs,
                     'loss':self.loss,
                     'val_loss':self.val_loss,
                     'loss_data':self.loss_data,     
                     'loss_eqns':self.loss_eqns,
                     'loss_conds':self.loss_conds,
                     'alpha':self.alpha,
                     'alpha_seq':self.alpha_seq,
                     'norm_paras':self.norm_paras})
        
        if self.nt_config.maxIter > 0:
            # reload the model to a single GPU
            self.model = self.loadNN()
            # L-BFGS training
            loopnum = np.fix(self.nt_config.maxIter/self.nt_config.stepIter)
            # starting
            for ii in np.arange(0,loopnum,1):
                # randomly select the training data
                if self.nt_config.batchSize > 0:
                    idx = np.random.choice(self.train_X.shape[0], self.nt_config.batchSize, replace=True)
                    self.cur_train_X = self.train_X[idx,:]
                    self.cur_train_Y = self.train_Y[idx,:]
                    
                elif self.nt_config.batchSize == 0:
                    self.cur_train_X = self.train_X
                    self.cur_train_Y = self.train_Y
                        
                # Transforms model into a function of its parameter
                func, params, names = tf_function_factory(self.model, self.loss_fn_all, self.cur_train_X, self.cur_train_Y)
                # Minimization
                res = minimize(func, 
                                params, 
                                method='L-BFGS-B',
                                options={'disp':None,
                                        'maxiter': self.nt_config.stepIter,
                                        'maxcor': 50,
                                        'maxls': 50,
                                        'gtol':1e-8,
                                        'eps':1e-8,
                                        'ftol': self.nt_config.tolFun},
                                callback= self.lbfgs_callback)  
        
            # save the model
            self.saveNN()
            # save the parameters
            sio.savemat('./weights/'+self.savename+'_paras.mat', 
                        {'batch_size':self.tf_config.batch_size,
                         'epochs':self.tf_config.epochs,
                         'loss':self.loss,
                         'loss_data':self.loss_data,     
                         'loss_eqns':self.loss_eqns,
                         'loss_conds':self.loss_conds,
                         'alpha':self.alpha,
                         'alpha_seq':self.alpha_seq,
                         'norm_paras':self.norm_paras})
        
        return history
    
        
    def predict(self, input_X):
        # prediction
        Y = self.model.predict(input_X, batch_size=8196*self.tf_config.gpus_number,
                               workers=4, use_multiprocessing=True)
        u = Y[:,0:1]
        v = Y[:,1:2]
        w = Y[:,2:3]
        p = Y[:,3:4]        
        # recover to real value
        u = u*self.norm_paras[1,4]+self.norm_paras[0,4]
        v = v*self.norm_paras[1,5]+self.norm_paras[0,5]
        w = w*self.norm_paras[1,6]+self.norm_paras[0,6]
        p = p*self.norm_paras[1,7]+self.norm_paras[0,7]
        
        return u, v, w, p
    
    
    
    def loadNN(self):
        weight_file = './weights/'+ self.savename
        model = self.build_model()
        model.load_weights(weight_file)
        
        return model
    

    
    def saveNN(self):
        self.model.save_weights('./weights/'+self.savename)
        
    
             
    def loadParas(self):
        """
        Returns
        -------
        norm_paras
        loadmat不需要转置
        """
        data = sio.loadmat('./weights/'+self.savename+'_paras.mat')
        norm_paras = data['norm_paras']
        return norm_paras
        
    

    def piecewise_scheduler(self, epoch):
        """
        piecewise learning rate decay
        """
        rate = np.floor(epoch/100)
        return self.tf_config.init_lr/(rate+1.0)
    
    

    def exponential_continuous_scheduler(self, epoch):
        """
        exponential learning rate decay
        """
        decay_rate = 0.98
        decay_epoch = 100
        lr = self.tf_config.init_lr * np.power(decay_rate,(epoch / decay_epoch))
        if lr < 1e-6:
            return 1e-6
        else:
            return lr 
        
        
        
        
    def exponential_staircase_scheduler(self, epoch):
        """
        exponential learning rate decay
        """
        decay_rate = 0.98
        decay_epoch = 100
        lr = self.tf_config.init_lr * np.power(decay_rate, np.floor(epoch / decay_epoch))
        if lr < 1e-6:
            return 1e-6
        else:
            return lr      
            
    


    def constant_scheduler(self, epoch):
        """
        constant learning rate decay
        """
        return self.tf_config.init_lr   
    
    
    
    
    
class NavierStokes2DPINNs(object):
    def __init__(self, hp, data, eqns, *conds):
        """
        data points:    [t,x,y,u,v,p]
        eqns points:    [t,x,y,0,0,0]
        conds:          other conditions like boundaries
            # ii=0, initial BC,             [t,x,y,u,v,p]
            # ii=1, Dirichlet BC,           [t,x,y,u,v,p]
            # ii=2, Neumann BC of ux,       [t,x,y,ux]
            # ii=3, Neumann BC of uy,       [t,x,y,uy]
            # ii=4, Neumann BC of vx,       [t,x,y,vx]
            # ii=5, Neumann BC of vy,       [t,x,y,vy]
            # ii=6, Neumann BC of px,       [t,x,y,px]
            # ii=7, Neumann BC of py,       [t,x,y,py]    
            
        Note: equation points should include the points of boundary conditions
        """
        # clear session
        tf.keras.backend.clear_session()
        # hp is the structure of hyper-parameters
        self.dtype = 'float32'
        self.layers = hp['layers']
        self.ExistModel = hp['ExistModel']
        self.map_name = hp['map_name']
        self.savename = hp['savename']
        self.Re = hp['Re']
        self.alpha = hp['alpha']   
        self.training = hp['train']
        # record the iteration number
        self.iternum = 0

        # Setting the optimizers with the hyper-parameters
        self.nt_config = Struct()
        self.nt_config.learningRate = hp["nt_lr"]
        self.nt_config.maxIter = hp["nt_max_iternum"]
        self.nt_config.stepIter = hp["nt_steps_per_loop"]
        self.nt_config.batchSize = hp["nt_batch_size"]
        self.nt_config.tolFun = 1.0 * np.finfo(float).eps
        
        self.tf_config = Struct()
        self.tf_config.epochs = hp['tf_epochs']
        self.tf_config.initial_epoch = hp['tf_initial_epoch']
        self.tf_config.steps_per_epoch = hp['tf_steps_per_epoch']
        self.tf_config.batch_size = hp['tf_batch_size']
        self.tf_config.init_lr = hp['tf_init_lr']

        # get the boundary conditions
        self.conds = self.get_conditions(*conds)
        
        # merge the inputs to generate training data
        self.train_X, self.train_Y = self.merge_inputs(data, eqns)          
        
        # Initialize the loss recording list
        self.loss_all = []
        self.val_loss = []
        self.loss_data = []
        self.loss_eqns = []
        self.loss_conds = []
        self.alpha_seq = []
        # alpha_seq is initialized by alpha
        self.alpha_seq.append(self.alpha)
        

        # Multi GPU or single GPU
        if _GPU_NUM > 1:
            strategy = tf.distribute.MirroredStrategy()
        elif _GPU_NUM == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            
        self.tf_config.gpus_number = strategy.num_replicas_in_sync 
        self.tf_config.global_batch_size = self.tf_config.batch_size*self.tf_config.gpus_number        
        # learning rate
        reduce_lr = LearningRateScheduler(self.exponential_staircase_scheduler,verbose=0)
        # callback
        self.call_back_list =[
                ModelCheckpoint(filepath='./weights/', 
                                monitor='loss', save_best_only=True, period=1000),
                reduce_lr,
                LossPrintingCallback()]
                #EarlyStopping(monitor='val_loss', min_delta = 0.05, patience=100)]
        # optimizer     
        self.optimizer = Adam(lr=self.tf_config.init_lr)  


        # multi-GPU
        if self.training:
            with strategy.scope():        
                if self.ExistModel == 0:
                    self.norm_paras = hp['norm_paras']
                    self.model = self.build_model()
                elif self.ExistModel == 1:
                    self.norm_paras = self.loadParas()  
                    self.model = self.loadNN()
                
                self.model.compile(optimizer=self.optimizer,\
                                   loss=self.loss_fn_all,\
                                   metrics=[self.loss_fn_data, self.loss_fn_eqns, self.loss_fn_conds])
                self.model.summary()
        # single-GPU     
        else:
            if self.ExistModel == 0:
                self.norm_paras = hp['norm_paras']
                self.model = self.build_model()
            elif self.ExistModel == 1:
                self.norm_paras = self.loadParas()  
                self.model = self.loadNN()
            self.model.summary()           
                
        

    
    
    def build_model(self):
        model = generator(self.layers, self.norm_paras, map_name=self.map_name)
     
        return model
    
 
    
    def get_conditions(self, *conds):
        """
        get the other (boundary) conditions from the inputs
        """
        conds_keys = ['init', 'dirichlet', 
                      'ux', 'uy',
                      'vx', 'vy',
                      'px', 'py']
        
        # default is no conditions
        conds_dict = dict.fromkeys(conds_keys, None)
    
        conds_num = len(conds)
        if conds_num > 0:
            # there are conditions
            for ii in np.arange(0,conds_num,1):
                # ii=0, initial BC,             [t,x,y,u,v,p]
                # ii=1, Dirichlet BC,           [t,x,y,u,v,p]
                # ii=2, Neumann BC of ux,       [t,x,y,ux]
                # ii=3, Neumann BC of uy,       [t,x,y,uy]
                # ii=4, Neumann BC of vx,       [t,x,y,vx]
                # ii=5, Neumann BC of vy,       [t,x,y,vy]
                # ii=6, Neumann BC of px,       [t,x,y,px]
                # ii=7, Neumann BC of py,       [t,x,y,py] 
                if conds[ii] is None:
                    conds_dict[conds_keys[ii]] = None
                else:
                    conds_dict[conds_keys[ii]] = conds[ii]
                
        return conds_dict
            
            
    
    def merge_inputs(self, data, eqns):
        """
        data points:    [t,x,y,u,v,p]
        eqns points:    [t,x,y]
        
        train_X:        [t,x,y]
        train_Y:        [t,x,y,u,v,p,flag0,flag1]
        flag0:          flag of data points
        flag1:          flalg of equations points
        
        Note: equation points include the points of boundary conditions
        """
        if data is None: 
            # there is only equation points, data points
            eqns_num = eqns.shape[0]
            # [t, x, y, u, v, p, flag0,flag1]
            if self.dtype=='float32':
                train_Y = np.zeros([eqns_num,8], dtype=np.float32)
            elif self.dtype=='float64':
                train_Y = np.zeros([eqns_num,8], dtype=np.float64) 
            train_Y[0:eqns_num,0:3] = eqns
            train_Y[0:eqns_num,7:8] = 1.0
        else:
            # there are both eqns points and data points
            data_num = data.shape[0]
            if self.dtype=='float32':
                train_Y = np.zeros([data_num,8], dtype=np.float32)
            elif self.dtype=='float64':
                train_Y = np.zeros([data_num,8], dtype=np.float64)             
            train_Y[0:data_num,0:6] = data
            train_Y[0:data_num,6:7] = 1.0
            # adding eqns points
            eqns_num = eqns.shape[0]
            # [t, x, y, u, v, p, flag0,flag1]
            if self.dtype=='float32':
                tmp = np.zeros([eqns_num,8], dtype=np.float32)
            elif self.dtype=='float64':
                tmp = np.zeros([eqns_num,8], dtype=np.float64) 
            tmp[0:eqns_num,0:3] = eqns
            train_Y = np.concatenate([train_Y, tmp], axis=0) 
            # all the points need to estimate the residual of equations
            train_Y[:,7:8] = 1.0

        train_X = train_Y[:,0:3]
        return train_X, train_Y
            
  
    
    def ns_eqns(self, X):
        """
        Returns
        -------
        residual of Navier-Stokes equations

        """    
        t = tf.convert_to_tensor(X[:,0:1], self.dtype)
        x = tf.convert_to_tensor(X[:,1:2], self.dtype)
        y = tf.convert_to_tensor(X[:,2:3], self.dtype)
        
        # Using the new GradientTape paradigm of TF2.0
        # persistent: multi-times gradint
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t)
            tape1.watch(x)
            tape1.watch(y)
                
            with tf.GradientTape(persistent=True) as tape2:
                # Watching gradients of t,x,y,z
                tape2.watch(t)
                tape2.watch(x)
                tape2.watch(y)
                # Packing together the inputs
                X = tf.stack([t[:,0],x[:,0],y[:,0]], axis=1)
                # Getting the prediction
                # Y = self.model(X, training=self.training)
                # Y: [u,v,p,um,vm,pm,ustd,vstd,pstd]
                Y = self.model(X)
                u = Y[:,0:1]
                v = Y[:,1:2]
                p = Y[:,2:3]
                # recover to real value
                u = u*self.norm_paras[1,3]+self.norm_paras[0,3]
                v = v*self.norm_paras[1,4]+self.norm_paras[0,4]
                p = p*self.norm_paras[1,5]+self.norm_paras[0,5]
            # first-order deriavative
            u_t = tape2.gradient(u, t)
            u_x = tape2.gradient(u, x)
            u_y = tape2.gradient(u, y)
            v_t = tape2.gradient(v, t)
            v_x = tape2.gradient(v, x)
            v_y = tape2.gradient(v, y)
            p_x = tape2.gradient(p, x)
            p_y = tape2.gradient(p, y)
        # first-order deriavative
        u_xx = tape1.gradient(u_x, x)
        u_yy = tape1.gradient(u_y, y)
        v_xx = tape1.gradient(v_x, x)
        v_yy = tape1.gradient(v_y, y)
            
        # Letting the tape go
        del tape1, tape2
        
        e1 = u_t + (u * u_x + v * u_y) + p_x - (1.0 / self.Re) * (u_xx + u_yy)
        e2 = v_t + (u * v_x + v * v_y) + p_y - (1.0 / self.Re) * (v_xx + v_yy)
        e3 = u_x + v_y
    
        # Buidling the PINNs
        return e1, e2, e3

  
      
    def get_gradient(self, X, flag):
        """
        Returns
        -------
        the gradient of u,v,p

        """    
        t = tf.convert_to_tensor(X[:,0:1], self.dtype)
        x = tf.convert_to_tensor(X[:,1:2], self.dtype)
        y = tf.convert_to_tensor(X[:,2:3], self.dtype)
        
        # Using the new GradientTape paradigm of TF2.0
        # persistent: multi-times gradint               
        with tf.GradientTape(persistent=True) as tape:
            # Watching gradients of t,x,y,z
            tape.watch(t)
            tape.watch(x)
            tape.watch(y)
            # Packing together the inputs
            X = tf.stack([t[:,0],x[:,0],y[:,0]], axis=1)
            # Getting the prediction
            # Y = self.model(X, training=self.training)
            Y = self.model(X)
            u = Y[:,0:1]
            v = Y[:,1:2]
            p = Y[:,2:3]
            # recover to real value
            u = u*self.norm_paras[1,3]+self.norm_paras[0,3]
            v = v*self.norm_paras[1,4]+self.norm_paras[0,4]
            p = p*self.norm_paras[1,5]+self.norm_paras[0,5]
        # first-order deriavative
        if flag.lower() == 'ux':
            g = tape.gradient(u, x)
        elif flag.lower() == 'uy':      
            g = tape.gradient(u, y)
        elif flag.lower() == 'vx':
            g = tape.gradient(v, x)
        elif flag.lower() == 'vy':
            g = tape.gradient(v, y)    
        elif flag.lower() == 'px':
            g = tape.gradient(p, x)    
        elif flag.lower() == 'py':
            g = tape.gradient(p, y)                
    
        # Letting the tape go
        del tape
        
        # Buidling the PINNs
        return g 
    
    
    
    def get_uvp(self, X):
        """
        get the output of the network
        The predict function is designed for performance in large scale inputs. 
        For small amount of inputs that fit in one batch, directly using
        __call__() is recommended for faster execution, e.g., model(x), 
        or model(x, training=False)

        """ 
        Xi = tf.convert_to_tensor(X[:,0:3], self.dtype)
        Y = self.model(Xi)
        u = Y[:,0:1]*self.norm_paras[1,3]+self.norm_paras[0,3]
        v = Y[:,1:2]*self.norm_paras[1,4]+self.norm_paras[0,4]
        p = Y[:,2:3]*self.norm_paras[1,5]+self.norm_paras[0,5]
        
        return u, v, p
        

    
    def loss_fn_all(self, Y_true, Y_pred):
        """
        自定义loss
        Y_true: [t, x, y, u, v, p, flag0, flag1]
        """
        # update the iteration number
        self.iternum = self.iternum+1 
        
        # loss of equations
        le = self.loss_fn_eqns(Y_true, Y_pred)
        # loss of data
        ld = self.loss_fn_data(Y_true, Y_pred)    
        # loss of boundary condtions
        lb = self.loss_fn_conds(Y_true, Y_pred)
        # total loss                          
        return self.alpha*le+ld+lb
 
    
 
    def loss_fn_eqns(self, Y_true, Y_pred):
        """
        return the loss of eqns and Neumann BC
        """   
        #flag for equations
        flag_data = Y_true[:,7:8]
        num_data = tf.reduce_sum(flag_data)+1.0
        # 首先计算方程点的loss
        X = Y_true[:,0:3]
        e1,e2,e3 = self.ns_eqns(X)  
        loss_eqns_e1 = tf.reduce_sum(tf.square(e1*flag_data))/num_data
        loss_eqns_e2 = tf.reduce_sum(tf.square(e2*flag_data))/num_data
        loss_eqns_e3 = tf.reduce_sum(tf.square(e3*flag_data))/num_data
        loss_eqns = loss_eqns_e1+loss_eqns_e2+loss_eqns_e3
        
        return loss_eqns
    
    
    
    def loss_fn_conds(self, Y_true, Y_pred):
        """
        return the loss of all the BCs
        # ii=0, initial BC,             [t,x,y,u,v,p]
        # ii=1, Dirichlet BC,           [t,x,y,u,v,p]
        # ii=2, Neumann BC of ux,       [t,x,y,ux]
        # ii=3, Neumann BC of uy,       [t,x,y,uy]
        # ii=4, Neumann BC of vx,       [t,x,y,vx]
        # ii=5, Neumann BC of vy,       [t,x,y,vy]
        # ii=6, Neumann BC of px,       [t,x,y,px]
        # ii=7, Neumann BC of py,       [t,x,y,py]   
        """ 
        batch_size = 1000
        loss = 0.0
        # iterate to estimate the loss of Bcs
        for key, val in self.conds.items():
            if (key == 'init') or (key == 'dirichlet'):
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    idx = np.random.choice(val.shape[0], batch_size, replace=True)
                    tmpX = val[idx,0:3]
                    # using the fluctuation to calculate the error
                    ut = val[idx,3:4]
                    vt = val[idx,4:5]
                    # pt = val[idx,5:6]
                    up, vp, pp = self.get_uvp(tmpX)
                    
                    tmpu = tf.reduce_mean(tf.square(ut-up))
                    tmpv = tf.reduce_mean(tf.square(vt-vp))
                    
                    loss = loss + tmpu + tmpv
            # for the conditions of gradients: ux, uy, vx, vy, px, pz
            else:
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    idx = np.random.choice(val.shape[0], batch_size, replace=True)
                    tmpX = val[idx,0:3]
                    # using the fluctuation to calculate the error
                    gt = val[idx,3:4]
                    gp = self.get_gradient(tmpX, key)
                    
                    tmp = tf.reduce_mean(tf.square(gt-gp))
                    
                    loss = loss + tmp      
                    
        return loss
    


    def loss_fn_data(self, Y_true, Y_pred):
        """
        return the loss of data
        """     
        # flag for data
        flag_data = Y_true[:,6:7]
        num_data = tf.reduce_sum(flag_data)+1.0
        
        # get the data value
        ut = Y_true[:,3:4]
        vt = Y_true[:,4:5]
        # pt = Y_true[:,5:6]
        up = Y_pred[:,0:1]*self.norm_paras[1,3]+self.norm_paras[0,3]
        vp = Y_pred[:,1:2]*self.norm_paras[1,4]+self.norm_paras[0,4]
        # pp = Y_pred[:,2:3]*self.norm_paras[1,5]+self.norm_paras[0,5]

        tmp = tf.square(ut-up)
        loss_data_u = tf.reduce_sum(tmp*flag_data)/num_data
        tmp = tf.square(vt-vp)
        loss_data_v = tf.reduce_sum(tmp*flag_data)/num_data
        
        loss_data = loss_data_u+loss_data_v        
        return loss_data 
  



    def lbfgs_callback(self, Xi):
        # Xi is for code running

        # iteration number
        self.iternum = self.iternum+1
        
        # prediction
        X = self.cur_train_Y[:,0:3]
        Y_pred = self.model.predict(X,batch_size=8192,
                               workers=4, use_multiprocessing=True)
        
        # loss of equations
        le = self.loss_fn_eqns(self.cur_train_Y, Y_pred)
        # loss of the data
        ld = self.loss_fn_data(self.cur_train_Y, Y_pred)    
        # loss of other conditions (BCs)
        lb = self.loss_fn_conds(self.cur_train_Y, Y_pred)
        # total loss                          
        loss =  self.alpha*le+ld+lb
    
        self.loss_data.append(ld)
        self.loss_eqns.append(le)  
        self.loss_conds.append(lb)
        self.loss.append(loss)
        if self.iternum % 10 == 0:        
            print('L-BGFS-B Iter=%05d: Loss: %.4e, loss_data: %.4e, loss_eqns: %.4e, loss_conds: %.4e' %
                      (self.iternum, loss, ld, le, lb))          
   
    
    
    def train(self):
        self.training = True        
        history = self.model.fit(self.train_X, self.train_Y, batch_size=self.tf_config.global_batch_size, 
                                 epochs=self.tf_config.epochs, 
                                 verbose=0, 
                                 callbacks=self.call_back_list,
                                 validation_split=0.1,
                                 shuffle=True,
                                 initial_epoch=self.tf_config.initial_epoch, 
                                 steps_per_epoch=self.tf_config.steps_per_epoch,
                                 workers=self.tf_config.gpus_number, use_multiprocessing=True)
        
        self.loss_data = history.history['loss_fn_data']
        self.loss_eqns = history.history['loss_fn_eqns']
        self.loss_conds = history.history['loss_fn_conds']
        self.loss = history.history['loss']
        self.val_loss = history.history['val_loss']
        
        # save model
        self.saveNN()  
        # save the parameters
        sio.savemat('./weights/'+self.savename+'_paras.mat', 
                    {'batch_size':self.tf_config.batch_size,
                     'epochs':self.tf_config.epochs,
                     'loss':self.loss,
                     'val_loss':self.val_loss,
                     'loss_data':self.loss_data,     
                     'loss_eqns':self.loss_eqns,
                     'loss_conds':self.loss_conds,
                     'alpha':self.alpha,
                     'alpha_seq':self.alpha_seq,
                     'norm_paras':self.norm_paras})
        
        if self.nt_config.maxIter > 0:
            # reload the model to a single GPU
            self.model = self.loadNN()
            # L-BFGS training
            loopnum = np.fix(self.nt_config.maxIter/self.nt_config.stepIter)
            # starting
            for ii in np.arange(0,loopnum,1):
                # randomly select the training data
                if self.nt_config.batchSize > 0:
                    idx = np.random.choice(self.train_X.shape[0], self.nt_config.batchSize, replace=True)
                    self.cur_train_X = self.train_X[idx,:]
                    self.cur_train_Y = self.train_Y[idx,:]
                    
                elif self.nt_config.batchSize == 0:
                    self.cur_train_X = self.train_X
                    self.cur_train_Y = self.train_Y
                        
                # Transforms model into a function of its parameter
                func, params, names = tf_function_factory(self.model, self.loss_fn_all, self.cur_train_X, self.cur_train_Y)
                # Minimization
                res = minimize(func, 
                                params, 
                                method='L-BFGS-B',
                                options={'disp':None,
                                        'maxiter': self.nt_config.stepIter,
                                        'maxcor': 50,
                                        'maxls': 50,
                                        'gtol':1e-8,
                                        'eps':1e-8,
                                        'ftol': self.nt_config.tolFun},
                                callback= self.lbfgs_callback)  
        
            # save the model
            self.saveNN()
            # save the parameters
            sio.savemat('./weights/'+self.savename+'_paras.mat', 
                        {'batch_size':self.tf_config.batch_size,
                         'epochs':self.tf_config.epochs,
                         'loss':self.loss,
                         'loss_data':self.loss_data,     
                         'loss_eqns':self.loss_eqns,
                         'loss_conds':self.loss_conds,
                         'alpha':self.alpha,
                         'alpha_seq':self.alpha_seq,
                         'norm_paras':self.norm_paras})
        
        return history
    
        
    def predict(self, input_X):
        # prediction  
        Y = self.model.predict(input_X, batch_size=8192*self.tf_config.gpus_number,
                               workers=4, use_multiprocessing=True)
        # Y = self.model(input_X, training=False)
        u = Y[:,0:1]
        v = Y[:,1:2]
        p = Y[:,2:3]  
        # recover to real value
        u = u*self.norm_paras[1,3]+self.norm_paras[0,3]
        v = v*self.norm_paras[1,4]+self.norm_paras[0,4]
        p = p*self.norm_paras[1,5]+self.norm_paras[0,5]
        
        return u, v, p
    
    
    
    def loadNN(self):
        weight_file = './weights/'+ self.savename
        model = self.build_model()
        model.load_weights(weight_file)
        
        return model
    

    
    def saveNN(self):
        self.model.save_weights('./weights/'+self.savename)
        
    
             
    def loadParas(self):
        """
        Returns
        -------
        norm_paras
        loadmat: no transpose
        """
        data = sio.loadmat('./weights/'+self.savename+'_paras.mat')
        norm_paras = data['norm_paras']
        return norm_paras
        
    

    def piecewise_scheduler(self, epoch):
        """
        piecewise learning rate decay
        """
        rate = np.floor(epoch/100)
        return self.tf_config.init_lr/(rate+1.0)
    
    

    def exponential_continuous_scheduler(self, epoch):
        """
        exponential learning rate decay
        """
        decay_rate = 0.98
        decay_epoch = 100
        lr = self.tf_config.init_lr * np.power(decay_rate,(epoch / decay_epoch))
        if lr < 1e-6:
            return 1e-6
        else:
            return lr 
        
        
        
        
    def exponential_staircase_scheduler(self, epoch):
        """
        exponential learning rate decay
        """
        decay_rate = 0.98
        decay_epoch = 100
        lr = self.tf_config.init_lr * np.power(decay_rate, np.floor(epoch / decay_epoch))
        if lr < 1e-6:
            return 1e-6
        else:
            return lr        
            
    


    def constant_scheduler(self, epoch):
        """
        constant learning rate decay
        """
        return self.tf_config.init_lr    
    
    
    
    
    
    

    
