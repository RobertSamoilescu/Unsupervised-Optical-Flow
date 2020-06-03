from __future__ import division
import torch
import torch.nn.functional as F
from asn1crypto.util import extended_date

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert (all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    pixel_coords = None
    return cam_coords * depth.unsqueeze(1)


def transform(world_coords, rot, tr):
    b, _, h, w = world_coords.size()
    world_coods_flat = world_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if rot is not None:
        pcoords = rot @ world_coods_flat
    else:
        pcoords = world_coods_flat

    if tr is not None:
        pcoords = pcoords + tr  # [B, 3, H*W]

    return pcoords.reshape(b, 3, h, w)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode, norm=True):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    if norm:
        # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        X_norm = 2 * (X / Z) / (w - 1) - 1
        Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
        pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    else:
        X = X / Z
        Y = Y / Z  # Idem [B, H*W]
        pixel_coords = torch.stack([X, Y], dim=2)

    return pixel_coords.reshape(b, h, w, 2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach() * 0 + 1, quat], dim=1)
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                  1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def forward_warp_old(img, depth, pose, intrinsics, extrinsics=None, rotation_mode='euler', padding_mode='zeros'):
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    # camera coordinates
    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]
    aux = torch.ones(batch_size, 1, img_height, img_width).double()
    cam_coords = torch.cat((cam_coords, aux), dim=1)  # [B,4,H,W]
    assert cam_coords.shape == (batch_size, 4, img_height, img_width), "Camera coordinates not [B, 4, H, W]"

    cam_coords = torch.transpose(cam_coords, 1, 2)
    cam_coords = torch.transpose(cam_coords, 2, 3)
    cam_coords = cam_coords.view(batch_size, img_height * img_width, -1)  # [B, HW, 4]
    cam_coords = torch.transpose(cam_coords, 1, 2)  # [B, 4, HW]
    assert cam_coords.shape == (batch_size, 4, img_width * img_height), "Camera coordinates not [B, 4, HW]"

    # compute real world position
    world_coords = cam_coords
    world_coords = world_coords.view(batch_size, 4, img_height, img_width)  # [B, 4, H, W]
    world_coords = world_coords[:, :3, :, :]

    # position matrix
    pose_mat = pose_vec2mat(pose, rotation_mode)  # [1, 3, 4]

    # projection matrix
    proj_matrix = intrinsics @ pose_mat
    rot, tr = proj_matrix[:, :, :3], proj_matrix[:, :, -1:]
    dst_pixel_coords = cam2pixel(
        world_coords, rot, tr, padding_mode, norm=False)  # [B,H,W,2]

    projected_img = torch.zeros_like(img)
    valid_points = torch.zeros(batch_size, img_height, img_width, dtype=torch.long)

    # define matrix of distances
    dists = 2 * torch.ones(batch_size, img_height, img_width, dtype=torch.double)
    z_buffer = 100. * torch.ones(batch_size, img_height, img_width, dtype=torch.double)

    for h in range(img_height):
        for w in range(img_width):
            x_coords, y_coords = dst_pixel_coords[:, h, w, 0], dst_pixel_coords[:, h, w, 1]
            z_coords = world_coords[:, h, w, 2]
            xs = [
                torch.floor(x_coords).type(torch.long),
                torch.ceil(x_coords).type(torch.long)
            ]
            ys = [
                torch.floor(y_coords).type(torch.long),
                torch.ceil(y_coords).type(torch.long)
            ]

            for x in xs:
                for y in ys:
                    mask = (x >= 0) * (x < img_width) * (y >= 0) * (y < img_height)
                    mx, my = x[mask], y[mask]
                    # print(mx.shape)

                    if len(mx) > 0 and len(my) > 0:
                        d = torch.pow(mx.type(torch.double) - x_coords[mask].type(torch.double), 2) \
                            + torch.pow(my.type(torch.double) - y_coords[mask].type(torch.double), 2)

                        cond = (d < dists[mask, my, mx]) & (z_buffer > z_coords) 
                        projected_img[mask, :, my, mx] = cond.type(torch.double).unsqueeze(dim=1) * img[mask, :, h, w] + \
                                                         (1 - cond.type(torch.double)).unsqueeze(dim=1) * projected_img[
                                                                                                          mask, :, my,
                                                                                                          mx]
                        valid_points[mask, my, mx] = cond.type(torch.long) + (1 - cond.type(torch.long)) * valid_points[
                            mask, my, mx]
                        dists[mask, my, mx] = cond.type(torch.double) * d + (1 - cond.type(torch.double)) * dists[
                            mask, my, mx]

    return projected_img, valid_points


def forward_warp(img, depth, pose, intrinsics, extrinsics=None, rotation_mode='euler', padding_mode='zeros'):
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    if extrinsics is not None:
        check_sizes(extrinsics, 'extrinsics', 'B44')

    batch_size, _, img_height, img_width = img.size()

    # camera coordinates
    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]
    aux = torch.ones(batch_size, 1, img_height, img_width).double()
    cam_coords = torch.cat((cam_coords, aux), dim=1)     # [B,4,H,W]
    assert cam_coords.shape == (batch_size, 4, img_height, img_width), "Camera coordinates not [B, 4, H, W]"

    # compute real world position
    if extrinsics is not None:
        extrinsics_inv = extrinsics.inverse()
        rot, tr = extrinsics_inv[:, :3, :3], extrinsics_inv[:, :3, -1:]
        world_coords = transform(cam_coords[:, :3, :, :], rot, tr)
    else:
        world_coords = cam_coords[:, :3, :, :]

    # position matrix
    pose_mat = pose_vec2mat(pose, rotation_mode)  # [1, 3, 4]
    rot, tr = pose_mat[:, :, :3], pose_mat[:, :, -1:]
    new_world_coords = transform(world_coords, rot, tr)

    # projection matrix
    if extrinsics is not None:
        proj_matrix = intrinsics @ extrinsics[:, :3, :]
    else:
        identity = torch.eye(4).unsqueeze(0).double()
        proj_matrix = intrinsics @ identity[:, :3, :]
    
    rot, tr = proj_matrix[:, :, :3], proj_matrix[:, :, -1:]
    dst_pixel_coords = cam2pixel(new_world_coords, rot, tr, padding_mode, norm=False)  # [B,H,W,2]
    projected_img = torch.zeros_like(img)
    valid_points = torch.zeros(batch_size, img_height, img_width, dtype=torch.long)

    # define matrix of distances
    dists = 2 * torch.ones(batch_size, img_height, img_width, dtype=torch.double)
    z_buffer = 1000 * torch.ones_like(dists)

    for h in range(img_height):
        for w in range(img_width):
            x_coords, y_coords = dst_pixel_coords[:, h, w, 0], dst_pixel_coords[:, h, w, 1]
            xs = [
                torch.floor(x_coords).type(torch.long),
                torch.ceil(x_coords).type(torch.long)
            ]
            ys = [
                torch.floor(y_coords).type(torch.long),
                torch.ceil(y_coords).type(torch.long)
            ]

            for x in xs:
                for y in ys:
                    mask = (x >= 0) * (x < img_width) * (y >= 0) * (y < img_height)
                    mx, my = x[mask], y[mask]
                    # print(mx.shape)

                    if len(mx) > 0 and len(my) > 0:
                        # distance between projection and a corner
                        d = torch.pow(mx.type(torch.double) - x_coords[mask].type(torch.double), 2) \
                            + torch.pow(my.type(torch.double) - y_coords[mask].type(torch.double), 2) 
                        # z component
                        dd = new_world_coords[mask, 2, h, w]

                        cond = (d < dists[mask, my, mx]) & (z_buffer[mask, my, mx] > dd)
                        projected_img[mask, :, my, mx] = cond.type(torch.double).unsqueeze(dim=1) * img[mask, :, h, w] + \
                                                         (1 - cond.type(torch.double)).unsqueeze(dim=1) * projected_img[mask, :, my, mx]
                        valid_points[mask, my, mx] = cond.type(torch.long) + \
                                (1 - cond.type(torch.long)) * valid_points[mask, my, mx]
                        dists[mask, my, mx] = cond.type(torch.double) * d + \
                                (1 - cond.type(torch.double)) * dists[mask, my, mx]
                        z_buffer[mask, my, mx] = cond.type(torch.double) * dd + \
                                (1 - cond.type(torch.double)) * z_buffer[mask, my, mx]
    return projected_img, valid_points

def inverse_warp(img, depth, pose, intrinsics, extrinsics=None, rotation_mode='euler', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        extrinisc: camera extrinisic matrix -- [B, 3, 4]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    if extrinsics is not None:
        aux = torch.tensor([[[0., 0., 0., 1.]]]).double()  # [1, 1, 4]
        aux = aux.repeat(batch_size, 1, 1)

        # append aux tensor
        aux_pose_mat = torch.cat((pose_mat, aux), dim=1)  # [B, 4, 4]
        aux_extrinsics = torch.cat((extrinsics, aux), dim=1)  # [B, 4, 4]

        # compute new pose_mat
        # pose_mat = extrinsics @ aux_pose_mat @ aux_extrinsics.inverse()  # [B, 3, 4]
        pose_mat = aux_extrinsics.inverse()[:, :3, :] @ aux_pose_mat @ aux_extrinsics

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords = cam2pixel(
        cam_coords, rot, tr, padding_mode)  # [B,H,W,2]

    projected_img = F.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points


def cam2pixel2(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2 * (X / Z) / (w - 1) - 1
    Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        # make sure that no point in warped image is a combinaison of im and gray
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w)


def inverse_warp2(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth = cam2pixel2(
        cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    projected_depth = F.grid_sample(
        ref_depth, src_pixel_coords, padding_mode=padding_mode).clamp(min=1e-3)

    return projected_img, valid_mask, projected_depth, computed_depth