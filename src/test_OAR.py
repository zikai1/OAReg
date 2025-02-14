import os
import open3d as o3d
import numpy as np
import glob
from tqdm import tqdm
import argparse
import time
import sys
sys.path.append("../")
import torch
from torch import nn
import pytorch3d
from pytorch3d.io import load_ply
from utils2.normalize_pointcloud import normalize_ply,normalize_ply_file
from utils2.LLR import barycenter_weights, barycenter_kneighbors_graph, local_linear_reconstruction
from utils2.loss_functions import correntropy_chamfer_distance
from model.model import Siren

torch.cuda.set_device(1)
DEVICE = 'cuda'






# deformation learning
def deform_point_cloud(model, xsrc=None, xtrg=None,
                 n_samples=10000, n_steps=200, sigma2=1.0, init_lr=1.0e-4,
                 LLR_weight=1.0e2, MCC_chamfer_weight=1.0e4,
                 LLR_n_neighbors=30, eval_every_nth_step=100, point_num=None):
  """
  Deform a point cloud using a neural network model.

  Parameters
  ----------
  model : torch.nn.Module
      The neural network model to use for deformation.
  xsrc : numpy.ndarray
      The source point cloud to deform.
  xtrg : numpy.ndarray, optional
      The target point cloud to match (used in MCC distance loss).
  n_samples : int, optional
      The number of points to sample for MCC distance loss (default is 10**4).
  n_steps : int, optional
      The number of optimization steps (default is 200).
  init_lr : float, optional
      The initial learning rate for the optimizer (default is 1.0e-4).
  LRR_weight : float, optional
      The weight for LRR loss (default is 1.0e2).
  MCC_chamfer_weight : float, optional
      The weight for chamfer distance loss (default is 1.0e4).
  LLR_n_neighbors: int, optional
      The number of neighbors to use for LRR loss (default is 30).
  eval_every_nth_step : int, optional
      The number of steps between evaluations (default is 100).
  point_num: int, optional
      The minimal number of the two input point clouds 
  """

  model = model.train()
  optm = torch.optim.Adam(model.parameters(), lr=init_lr)# optimizer
  schedm = torch.optim.lr_scheduler.ReduceLROnPlateau(optm, verbose=True, patience=1)# lr


  MCC_chamfer_loss_total = 0
  LLR_loss_total = 0
  total_loss = 0
  n_r = 0

  # Downsampling
  n_samples=5000
  if n_samples>point_num:
      n_samples=point_num
      

  for i in range(0, n_steps):
    xbatch_src=xsrc[np.random.choice(len(xsrc), n_samples, replace=False)]
    xbatch_trg=xtrg[np.random.choice(len(xtrg), n_samples, replace=False)]
    xbatch_deformed = xbatch_src + model(xbatch_src)

    loss = 0

    # LLR loss
    LLR_loss = LLR_weight*local_linear_reconstruction(xbatch_src, xbatch_deformed, n_neighbors=LLR_n_neighbors)
    loss += LLR_loss
    LLR_loss_total += float(LLR_loss)


    # MCC
    MCC_loss=correntropy_chamfer_distance(xbatch_deformed.unsqueeze(0),xbatch_trg.unsqueeze(0),sigma2=sigma2)
    MCC_chamfer_loss = MCC_chamfer_weight*MCC_loss
    loss += MCC_chamfer_loss
    MCC_chamfer_loss_total += float(MCC_chamfer_loss)
       

    total_loss += float(loss)
    n_r += 1

    optm.zero_grad()
    loss.backward()
    optm.step()

    # Evaluate the training results
    if i % eval_every_nth_step == 0:

      LLR_loss_total /= n_r
      MCC_chamfer_loss_total /= n_r
      total_loss /= n_r

      schedm.step(float(total_loss))
      


      LLR_loss_total = 0
      MCC_chamfer_loss_total = 0
      total_loss = 0
      n_r = 0

  LLR_loss_total /= n_r
  MCC_chamfer_loss_total /= n_r
  total_loss /= n_r




def MCC_registration(name=None,xsrc=None, xtrg=None, 
                     target_normal_scale=None,target_normal_center=None,
                     n_steps=200,
                     sigma2=1.0,
                     LLR_n_neighbors=30,
                     LLR_WEIGHT=1.0e2,
                     MCC_chamfer_WEIGHT=1.0e4,
                     data_deformed=None,
                     point_num=None):
    

#  define the deformation model
    model = Siren(in_features=3,
                    hidden_features=128,
                    hidden_layers=3,
                    out_features=3, outermost_linear=True,
                    first_omega_0=30, hidden_omega_0=30.).to(DEVICE).train()
    
    deform_point_cloud(model,
            xsrc=xsrc, xtrg=xtrg,
            init_lr=1.0e-4,
            n_steps=n_steps,
            sigma2=sigma2,
            LLR_n_neighbors=LLR_n_neighbors,
            LLR_weight=LLR_WEIGHT,
            MCC_chamfer_weight=MCC_chamfer_WEIGHT,
            point_num=point_num)
    
    
    model.eval()
    vpred = xsrc + model(xsrc).detach().clone()

    vpred_save=vpred.cpu().numpy()

    vpred_save_denormalize=target_normal_scale*vpred_save+target_normal_center


    pcd_deformed=o3d.geometry.PointCloud()
    pcd_deformed.points=o3d.utility.Vector3dVector(vpred_save_denormalize)


    save_pc_name=data_deformed+name+str(LLR_n_neighbors)+'.ply'

    o3d.io.write_point_cloud(save_pc_name,pcd_deformed)



if __name__=='__main__':
    SOURCE_MESH_PATH ="../data/source/"
    TARGET_MESH_PATH ="../data/target/"
    
    SOURCE_NORM_PATH ="../data/source_norm_data/"
    TARGET_NORM_PATH ="../data/target_norm_data/"

    data_deformed="../data/save_deformed/"


    if not os.path.exists(data_deformed): 
        os.mkdir(data_deformed) 

    source_files=os.listdir(SOURCE_MESH_PATH)
    target_files=os.listdir(TARGET_MESH_PATH)

    source_files.sort(key=lambda x:int(x[-6:-4]))
    target_files.sort(key=lambda x:int(x[-6:-4]))

    # Start registration  
    for i in range(0,len(source_files)):
        
        # Read point clouds 
        # Source
        source_data_name=source_files[i]
        print("source:",source_data_name)
        source_data=os.path.join(SOURCE_MESH_PATH,source_data_name)

        # Target
        target_data_name=target_files[i]
        print("target:",target_data_name)
        target_data=os.path.join(TARGET_MESH_PATH,target_data_name)

        # Normalize the input point clouds
        src_normalized_ply, src_normal_center, src_normal_scale = normalize_ply_file(source_data)
        tgt_normalized_ply, tgt_normal_center, tgt_normal_scale = normalize_ply_file(target_data)
        

        # Get points 
        src_points = np.asarray(src_normalized_ply.points,dtype=np.float32)
        tgt_points = np.asarray(tgt_normalized_ply.points,dtype=np.float32)

        src_points = torch.from_numpy(src_points).to(DEVICE)
        tgt_points = torch.from_numpy(tgt_points).to(DEVICE)
 
        src_point_num=src_points.shape[0]
        tgt_point_num=tgt_points.shape[0]


        if src_point_num>tgt_point_num:
            point_num=tgt_point_num
        else:
            point_num=src_point_num

        # Iterative optimization for registration
        MCC_registration(name=source_data_name,
                    xsrc=src_points, xtrg=tgt_points,
                    target_normal_scale=tgt_normal_scale,target_normal_center=tgt_normal_center,
                    n_steps=200,
                    sigma2=1.0,
                    LLR_n_neighbors=30,
                    LLR_WEIGHT=1.0e2,
                    MCC_chamfer_WEIGHT=1.0e4,
                    data_deformed=data_deformed,
                    point_num=point_num)

        print("**************************")
