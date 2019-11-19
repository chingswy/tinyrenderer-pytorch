import cv2
import numpy as np
import torch
# from gl.triangle import triangle
from gl.gl import viewport, lookat
from obj_io import obj_load

def triangle(t, A, B, C, light_here):
    global coords, texture, image, zbuffer
    (v0, uv0), (v1, uv1), (v2, uv2) = A, B, C
    device = v0.device
    # Barycentric coordinates of points inside the triangle bounding box
    # t = np.linalg.inv([[v0[0],v1[0],v2[0]], [v0[1],v1[1],v2[1]], [1,1,1]])
    xmin = int(max(0,              min(v0[0], v1[0], v2[0])))
    xmax = int(min(image.shape[1], max(v0[0], v1[0], v2[0])+1))
    ymin = int(max(0,              min(v0[1], v1[1], v2[1])))
    ymax = int(min(image.shape[0], max(v0[1], v1[1], v2[1])+1))
    P = coords[:, xmin:xmax, ymin:ymax].reshape(2,-1).float()
    B = t @ torch.cat((P, torch.ones(1, P.shape[1], \
        device=device)), dim=0)
    # Cartesian coordinates of points inside the triangle
    I = torch.nonzero(torch.all(B >= 0, dim=0))
    
    X, Y, Z = P[0,I], P[1,I], v0[2]*B[0,I] + v1[2]*B[1,I] + v2[2]*B[2,I]
    X, Y = X.long(), Y.long()
    # Texture coordinates of points inside the triangle
    U = (    (uv0[0]*B[0,I] + uv1[0]*B[1,I] + uv2[0]*B[2,I]))*(texture.shape[0]-1)
    V = (1.0-(uv0[1]*B[0,I] + uv1[1]*B[1,I] + uv2[1]*B[2,I]))*(texture.shape[1]-1)
    C = texture[V.long(), U.long()]
    # Z-Buffer test
    if I.shape[0] > 0:
        pass
        # import ipdb; ipdb.set_trace()        
    I = torch.nonzero(zbuffer[Y,X] < Z)[:,0]
    X, Y, Z, C = X[I], Y[I], Z[I], C[I]
    zbuffer[Y, X] = Z
    image[Y, X] = C * light_here


if __name__ == '__main__':
    import time
    device = torch.device('cpu')
    width, height = 1024, 1024
    light         = torch.Tensor([[0,0,-1]]).to(device)
    eye           = torch.Tensor([[1,1,3]]).to(device)
    center        = torch.Tensor([[0,0,0]]).to(device)
    up            = torch.Tensor([[0,1,0]]).to(device)

    image = torch.zeros((height,width,4), device=device)
    zbuffer = -1000*torch.ones((height,width), device=device)

    coords = torch.Tensor(np.mgrid[0:width, 0:height]).to(device).long()

    V, UV, Vi, UVi = obj_load("obj/african_head/african_head.obj", device)

    texture = cv2.imread("obj/african_head/african_head_diffuse.png", cv2.IMREAD_UNCHANGED)
    if texture.shape[2] == 3: #comple one channel
        texture = np.dstack((texture, 255*np.ones((texture.shape[0], texture.shape[1], 1), dtype='uint8')))
    import matplotlib.pyplot as plt
    plt.imshow(texture)
    plt.show()
    texture = torch.Tensor(texture).to(device)

    viewport = viewport(32, 32, width-64, height-64, 1000, device=device)
    modelview = lookat(eye, center, up)

    start = time.time()
    Vh = torch.cat([V, torch.ones(len(V), 1, device=device)], dim=1) # Homogenous coordinates
    V = Vh @ modelview.t()           # World coordinates
    Vs = V @ viewport.t()            # Screen coordinates
    V, Vs = V[:,:3],  Vs[:,:3]     # Back to cartesian coordinates
    V, Vs, UV = V[Vi], Vs[Vi], UV[UVi]
    # Pre-compute tri-linear coordinates
    T = torch.transpose(Vs, 1, 2).clone()
    T[:, 2,:] = 1
    T = torch.inverse(T)
    # Pre-compute normal vectors and intensity
    N = torch.cross(V[:,2]-V[:,0], V[:,1]-V[:,0])
    N = N / torch.norm(N,dim=1, keepdim=True)
    I = (N*light).sum(dim=1)
    light_point = torch.ones(1, 1, texture.shape[2], device=device)
    plt.ion()
    for i in torch.nonzero(I>=0)[:,0]:
        (vs0, vs1, vs2), (uv0, uv1, uv2) = Vs[i], UV[i]
        light_point[:, :, :3] = I[i]
        triangle(T[i], (vs0,uv0), (vs1,uv1), (vs2,uv2), light_point)

        # plt.imshow(img[:,:,:3])
        # plt.pause(0.01)
    end = time.time()
    plt.ioff()
    img = image.detach().cpu().numpy().astype('uint8')
    plt.imshow(img[:,:,[2,1,0,3]])
    plt.show()
    print("Rendering time: {}".format(end-start))
