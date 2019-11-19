import torch

def viewport(x, y, w, h, d=0, device=torch.device('cpu')):
    Viewport = torch.Tensor(
        [[w/2, 0, 0, x+w/2],
        [0, h/2, 0, y+h/2],
        [0, 0, d/2,   d/2],
        [0, 0, 0,       1]])
    Viewport = Viewport.to(device)
    return Viewport

def lookat(eye, center, up):
    device = eye.device
    normalize = lambda x: x/x.norm()
    M = torch.eye(4, device=device)
    z = normalize(eye - center)
    x = normalize(torch.cross(up,z))
    y = normalize(torch.cross(z,x))
    M[0,:3], M[1,:3], M[2,:3], M[:3,3] = x, y, z, -center
    return M

def triangle(t, A, B, C, intensity):
    global coords, texture, image, zbuffer
    (v0, uv0), (v1, uv1), (v2, uv2) = A, B, C

    # Barycentric coordinates of points inside the triangle bounding box
    # t = np.linalg.inv([[v0[0],v1[0],v2[0]], [v0[1],v1[1],v2[1]], [1,1,1]])
    xmin = int(max(0,              min(v0[0], v1[0], v2[0])))
    xmax = int(min(image.shape[1], max(v0[0], v1[0], v2[0])+1))
    ymin = int(max(0,              min(v0[1], v1[1], v2[1])))
    ymax = int(min(image.shape[0], max(v0[1], v1[1], v2[1])+1))
    P = coords[:, xmin:xmax, ymin:ymax].reshape(2,-1)
    B = np.dot(t, np.vstack((P, np.ones((1, P.shape[1])))))

    # Cartesian coordinates of points inside the triangle
    I = np.argwhere(np.all(B >= 0, axis=0))
    X, Y, Z = P[0,I], P[1,I], v0[2]*B[0,I] + v1[2]*B[1,I] + v2[2]*B[2,I]

    # Texture coordinates of points inside the triangle
    U = (    (uv0[0]*B[0,I] + uv1[0]*B[1,I] + uv2[0]*B[2,I]))*(texture.shape[0]-1)
    V = (1.0-(uv0[1]*B[0,I] + uv1[1]*B[1,I] + uv2[1]*B[2,I]))*(texture.shape[1]-1)
    C = texture[V.astype(int), U.astype(int)]

    # Z-Buffer test
    I = np.argwhere(zbuffer[Y,X] < Z)[:,0]
    X, Y, Z, C = X[I], Y[I], Z[I], C[I]
    zbuffer[Y, X] = Z
    image[Y, X] = C * (intensity, intensity, intensity, 1)
