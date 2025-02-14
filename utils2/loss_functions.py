import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss.chamfer import chamfer_distance,_handle_pointcloud_input
from pytorch3d.loss.chamfer import _handle_pointcloud_input




def correntropy_chamfer_distance(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        norm=1,
        sigma2=1.0
):
    """
    Correntropy Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    N, P1, D = x.shape
    P2 = y.shape[1]

    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1] 

    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]


    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)
    trunc_x=0.2   
    trunc_y=0.2   
    x_mask[cham_x >= trunc_x] = True
    y_mask[cham_y >= trunc_y] = True
    cham_x[x_mask] = 0.0
    cham_y[y_mask] = 0.0

    #  correntropy criterion
    exp_cham_x=torch.exp(-1*cham_x/(sigma2))
    exp_cham_y=torch.exp(-1*cham_y/(sigma2))

    cham_x=exp_cham_x 
    cham_y=exp_cham_y



    cham_x = cham_x.sum(1)
    cham_y = cham_y.sum(1)

    cham_x /= x_lengths
    cham_y /= y_lengths 
        

    cham_x = cham_x.sum()
    cham_y = cham_y.sum()


    cham_dist = -1.0*(cham_x + cham_y)

    return cham_dist