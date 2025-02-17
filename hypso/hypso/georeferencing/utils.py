import numpy as np
import numpy.typing as npt
import math as m




# Code copied from https://github.com/NTNU-SmallSat-Lab/ground_systems/blob/4c41925f5fbf6161a60d273cfee82bce22cfeffc/scripts/capture_processing/adcs-tm-strip.py#L152

def check_star_tracker_orientation(adcs: dict):

    adcs_samples=adcs['adcssamples']
    quaternion_s=adcs['quaternion_s']
    quaternion_x=adcs['quaternion_x']
    quaternion_y=adcs['quaternion_y']
    quaternion_z=adcs['quaternion_z']
    velocity_x=adcs['velocity_x']
    velocity_y=adcs['velocity_y']
    velocity_z=adcs['velocity_z']

    quat = np.empty((adcs_samples,4))
    quat[:,0] = quaternion_s
    quat[:,1] = quaternion_x
    quat[:,2] = quaternion_y
    quat[:,3] = quaternion_z

    vel = np.empty((adcs_samples,3))
    vel[:,0] = velocity_x
    vel[:,1] = velocity_y
    vel[:,2] = velocity_z

    st_vel_angles = np.zeros([adcs_samples,1])

    for i in range(adcs_samples):
        st_vel_angles[i] = compute_st_vel_angles(quat[i,:], vel[i,:])

    if st_vel_angles.mean() > 90.0:
        # was pointing away from velocity direction --> don't flip 
        flip = False
    else: 
        # was pointing in velocity direction --> do flip
        flip = True

    return flip


def compute_st_vel_angles(quat, vel):

    # Checks which direction the star tracker is pointing relative to velocity vector
    # Return true if the star tracker is pointing in velocity direction
    # Return false if the star tracker is pointing away from velocity direction

    # code from https://github.com/NTNU-SmallSat-Lab/ground_systems/blob/8e73a02055e3ebf935f306ac927c12674cb434dc/scripts/capture_processing/adcs-tm-strip.py#L72
    # code from https://github.com/NTNU-SmallSat-Lab/hypso-package/blob/bf9b3464137211278584ad0064afddf3a01d0c11/hypso/georeference/georef/geometric.py#L73

    body_x_body = np.array([1.0, 0.0, 0.0]) # this is star tracker direction
    #body_z_body = np.array([0.0, 0.0, 1.0])

    '''
    quat must be a four element list of numbers or 4 element nump array
    returns a 3x3 numpy array containing the rotation matrix
    '''
    mag = m.sqrt(quat[0]**2 + quat[1]**2 + quat[2]**2 + quat[3]**2)
    quat[0] /= mag
    quat[1] /= mag
    quat[2] /= mag
    quat[3] /= mag
 
    w2 = quat[0]*quat[0]
    x2 = quat[1]*quat[1]
    y2 = quat[2]*quat[2]
    z2 = quat[3]*quat[3]

    wx = quat[0]*quat[1]
    wy = quat[0]*quat[2]
    wz = quat[0]*quat[3]
    xy = quat[1]*quat[2]
    xz = quat[1]*quat[3]
    zy = quat[3]*quat[2]

    mat = np.zeros([3,3])

    mat[0,0] = w2+x2-y2-z2
    mat[1,0] = 2.0*(xy+wz)
    mat[2,0] = 2.0*(xz-wy)
    mat[0,1] = 2.0*(xy-wz)
    mat[1,1] = w2-x2+y2-z2
    mat[2,1] = 2.0*(zy+wx)
    mat[0,2] = 2.0*(xz+wy)
    mat[1,2] = 2.0*(zy-wx)
    mat[2,2] = w2-x2-y2+z2
    body_x_teme = np.matmul(mat,body_x_body)
    #body_z_teme = np.matmul(mat,body_z_body)
    
    vellen = m.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
    cos_vel_angle = (vel[0]*body_x_teme[0] + vel[1]*body_x_teme[1] + vel[2]*body_x_teme[2]) / vellen
    velocity_angle = m.acos(cos_vel_angle)*180.0/m.pi

    return velocity_angle


def compute_polynomial_transform(X, Y, lat_coefficients, lon_coefficients):
    
    ## Example usage:
    #for Y in range(0, image_height):
    #    for X in range(0, image_width):
    #        lon, lat = self.compute_polynomial_transform(Y, X, lat_coefficients, lon_coefficients)
    #        lats[Y,X] = lat
    #        lons[Y,X] = lon


    #X = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))

    #x.T = [a00 a10 a11 a20 a21 a22 ... ann
    #   b00 b10 b11 b20 b21 b22 ... bnn c3]

    #X = (( a_00 * x**(0 - 0) * y**0 ))
    #(( a_10 * x**(1 - 0) * y**0 ))  +  (( a_11 * x**(1 - 1) * y**1 ))
    #(( a_20 * x**(2 - 0) * y**0 ))  +  (( a_21 * x**(2 - 1) * y**1 )) 
    #                                +  (( a_22 * x**(2 - 2) * y**2 ))

    c = lat_coefficients
    lat = c[0] + c[1]*X + c[2]*Y + c[3]*X**2 + c[4]*X*Y + c[5]*Y**2

    c = lon_coefficients
    lon = c[0] + c[1]*X + c[2]*Y + c[3]*X**2 + c[4]*X*Y + c[5]*Y**2

    return (lat, lon)
