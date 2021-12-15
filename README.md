# PINN-TF2.1
train.py is used for training the PINN network

predict.py is used for predicting the fields of velocity and pressure

Three examples are provided:
1. Train_NS3D_PIV(): using a 2D2C homogeneous and isotropic dataset to train a 3D3C PINN network;
2. Train_Hemi3D_PIV(): using the Tomo-PIV data in the three-dimensional wake flow of a hemisphere to train a 3D3C PINN network;
3. Train_NSFlow2D(): using a 2D2C Taylor’s decaying vortices to train a 2D2C PINN network.
