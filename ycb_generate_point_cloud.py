import os
import numpy as np
import h5py as h5
from imageio import imread
import math
import struct
import tqdm
import open3d as o3

# Parameters
ycb_data_folder = "./ycb//"	# Folder that contains the ycb data.
viewpoint_cameras = ["NP1", "NP2", "NP3", "NP4", "NP5"]	# Camera which the viewpoint will be generated.
viewpoint_angles = [str(i) for i in range(0, 360, 3)] # Relative angle of the object w.r.t the camera (angle of the turntable).

# Objects
f = open(os.path.join(ycb_data_folder, 'objects.txt'))
object_list = f.readlines()
f.close()
print('Get {0} objects in total.'.format(len(object_list)))
for i, object_name in enumerate(object_list):
    object_list[i] = object_name.rstrip()

# object_list = ['001_chips_can']
# object_list = object_list[1:]
# object_list = object_list[3:20]
# object_list = object_list[20:38]
# object_list = object_list[38:56]
# object_list = object_list[56:74]
# object_list = object_list[74:]


def im2col(im, psize):
    n_channels = 1 if len(im.shape) == 2 else im.shape[0]
    (n_channels, rows, cols) = (1,) * (3 - len(im.shape)) + im.shape

    im_pad = np.zeros((n_channels,
                       int(math.ceil(1.0 * rows / psize) * psize),
                       int(math.ceil(1.0 * cols / psize) * psize)))
    im_pad[:, 0:rows, 0:cols] = im

    final = np.zeros((im_pad.shape[1], im_pad.shape[2], n_channels,
                      psize, psize))
    for c in range(n_channels):
        for x in range(psize):
            for y in range(psize):
                im_shift = np.vstack(
                    (im_pad[c, x:], im_pad[c, :x]))
                im_shift = np.column_stack(
                    (im_shift[:, y:], im_shift[:, :y]))
                final[x::psize, y::psize, c] = np.swapaxes(
                    im_shift.reshape(int(im_pad.shape[1] / psize), psize,
                                     int(im_pad.shape[2] / psize), psize), 1, 2)

    return np.squeeze(final[0:rows - psize + 1, 0:cols - psize + 1])

def filterDiscontinuities(depthMap):
    filt_size = 7
    thresh = 1000

    # Ensure that filter sizes are okay
    assert filt_size % 2 == 1, "Can only use odd filter sizes."

    # Compute discontinuities
    offset = int((filt_size - 1) / 2)
    patches = 1.0 * im2col(depthMap, filt_size)
    mids = patches[:, :, offset, offset]
    mins = np.min(patches, axis=(2, 3))
    maxes = np.max(patches, axis=(2, 3))

    discont = np.maximum(np.abs(mins - mids),
                         np.abs(maxes - mids))
    mark = discont > thresh

    # Account for offsets
    final_mark = np.zeros((480, 640), dtype=np.uint16)
    final_mark[offset:offset + mark.shape[0],
               offset:offset + mark.shape[1]] = mark

    return depthMap * (1 - final_mark)

def registerDepthMap(unregisteredDepthMap,
                     rgbImage,
                     depthK,
                     rgbK,
                     H_RGBFromDepth):


    unregisteredHeight = unregisteredDepthMap.shape[0]
    unregisteredWidth = unregisteredDepthMap.shape[1]

    registeredHeight = rgbImage.shape[0]
    registeredWidth = rgbImage.shape[1]

    registeredDepthMap = np.zeros((registeredHeight, registeredWidth))

    xyzDepth = np.empty((4,1))
    xyzRGB = np.empty((4,1))

    # Ensure that the last value is 1 (homogeneous coordinates)
    xyzDepth[3] = 1

    invDepthFx = 1.0 / depthK[0,0]
    invDepthFy = 1.0 / depthK[1,1]
    depthCx = depthK[0,2]
    depthCy = depthK[1,2]

    rgbFx = rgbK[0,0]
    rgbFy = rgbK[1,1]
    rgbCx = rgbK[0,2]
    rgbCy = rgbK[1,2]

    undistorted = np.empty(2)
    for v in range(unregisteredHeight):
      for u in range(unregisteredWidth):

            depth = unregisteredDepthMap[v,u]
            if depth == 0:
                continue

            xyzDepth[0] = ((u - depthCx) * depth) * invDepthFx
            xyzDepth[1] = ((v - depthCy) * depth) * invDepthFy
            xyzDepth[2] = depth

            xyzRGB[0] = (H_RGBFromDepth[0,0] * xyzDepth[0] +
                         H_RGBFromDepth[0,1] * xyzDepth[1] +
                         H_RGBFromDepth[0,2] * xyzDepth[2] +
                         H_RGBFromDepth[0,3])
            xyzRGB[1] = (H_RGBFromDepth[1,0] * xyzDepth[0] +
                         H_RGBFromDepth[1,1] * xyzDepth[1] +
                         H_RGBFromDepth[1,2] * xyzDepth[2] +
                         H_RGBFromDepth[1,3])
            xyzRGB[2] = (H_RGBFromDepth[2,0] * xyzDepth[0] +
                         H_RGBFromDepth[2,1] * xyzDepth[1] +
                         H_RGBFromDepth[2,2] * xyzDepth[2] +
                         H_RGBFromDepth[2,3])

            invRGB_Z  = 1.0 / xyzRGB[2]
            undistorted[0] = (rgbFx * xyzRGB[0]) * invRGB_Z + rgbCx
            undistorted[1] = (rgbFy * xyzRGB[1]) * invRGB_Z + rgbCy

            uRGB = int(undistorted[0] + 0.5)
            vRGB = int(undistorted[1] + 0.5)

            if (uRGB < 0 or uRGB >= registeredWidth) or (vRGB < 0 or vRGB >= registeredHeight):
                continue

            registeredDepth = xyzRGB[2]
            if registeredDepth > registeredDepthMap[vRGB,uRGB]:
                registeredDepthMap[vRGB,uRGB] = registeredDepth

    return registeredDepthMap

def registeredDepthMapToPointCloud(depthMap, rgbImage, rgbK, organized=True, mask=None):
    rgbCx = rgbK[0,2]
    rgbCy = rgbK[1,2]
    invRGBFx = 1.0/rgbK[0,0]
    invRGBFy = 1.0/rgbK[1,1]

    height = depthMap.shape[0]
    width = depthMap.shape[1]

    if organized:
      cloud = np.empty((height, width, 6), dtype=np.float)
    else:
      cloud = np.empty((1, height*width, 6), dtype=np.float)

    goodPointsCount = 0
    for v in range(height):
        for u in range(width):

            depth = depthMap[v,u]

            if organized:
              row = v
              col = u
            else:
              row = 0
              col = goodPointsCount

            if depth <= 0 or (mask is not None and mask[v,u] > 0):
                if organized:
                    if depth <= 0:
                       cloud[row,col,0] = float('nan')
                       cloud[row,col,1] = float('nan')
                       cloud[row,col,2] = float('nan')
                       cloud[row,col,3] = 0
                       cloud[row,col,4] = 0
                       cloud[row,col,5] = 0
                continue

            cloud[row,col,0] = (u - rgbCx) * depth * invRGBFx
            cloud[row,col,1] = (v - rgbCy) * depth * invRGBFy
            cloud[row,col,2] = depth
            cloud[row,col,3] = rgbImage[v,u,0]
            cloud[row,col,4] = rgbImage[v,u,1]
            cloud[row,col,5] = rgbImage[v,u,2]
            if not organized:
              goodPointsCount += 1

    if not organized:
      cloud = cloud[:,:goodPointsCount,:]

    return cloud


def writePLY(filename, cloud, faces=[]):
    if len(cloud.shape) != 3:
        print("Expected pointCloud to have 3 dimensions. Got %d instead" % len(cloud.shape))
        return

    color = True if cloud.shape[2] == 6 else False
    num_points = cloud.shape[0]*cloud.shape[1]

    header_lines = [
        'ply',
        'format ascii 1.0',
        'element vertex %d' % num_points,
        'property float x',
        'property float y',
        'property float z',
        ]
    if color:
        header_lines.extend([
        'property uchar diffuse_red',
        'property uchar diffuse_green',
        'property uchar diffuse_blue',
        ])
    if faces != None:
        header_lines.extend([
        'element face %d' % len(faces),
        'property list uchar int vertex_indices'
        ])

    header_lines.extend([
      'end_header',
      ])

    f = open(filename, 'w+')
    f.write('\n'.join(header_lines))
    f.write('\n')

    lines = []
    for i in range(cloud.shape[0]):
        for j in range(cloud.shape[1]):
            if color:
                lines.append('%s %s %s %d %d %d' % tuple(cloud[i, j, :].tolist()))
            else:
                lines.append('%s %s %s' % tuple(cloud[i, j, :].tolist()))

    for face in faces:
        lines.append(('%d' + ' %d'*len(face)) % tuple([len(face)] + list(face)))

    f.write('\n'.join(lines) + '\n')
    f.close()

def writePCD(pointCloud, filename, ascii=False):
    if len(pointCloud.shape) != 3:
      print("Expected pointCloud to have 3 dimensions. Got %d instead" % len(pointCloud.shape))
      return
    with open(filename, 'w') as f:
        height = pointCloud.shape[0]
        width = pointCloud.shape[1]
        f.write("# .PCD v.7 - Point Cloud Data file format\n")
        f.write("VERSION .7\n")
        if pointCloud.shape[2] == 3:
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
        else:
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
        f.write("WIDTH %d\n" % width)
        f.write("HEIGHT %d\n" % height)
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write("POINTS %d\n" % (height * width))
        if ascii:
          f.write("DATA ascii\n")
          for row in range(height):
            for col in range(width):
                if pointCloud.shape[2] == 3:
                    f.write("%f %f %f\n" % tuple(pointCloud[row, col, :]))
                else:
                    f.write("%f %f %f" % tuple(pointCloud[row, col, :3]))
                    r = int(pointCloud[row, col, 3])
                    g = int(pointCloud[row, col, 4])
                    b = int(pointCloud[row, col, 5])
                    rgb_int = (r << 16) | (g << 8) | b
                    packed = struct.pack('i', rgb_int)
                    rgb = struct.unpack('f', packed)[0]
                    f.write(" %.12e\n" % rgb)
        else:
          f.write("DATA binary\n")
          if pointCloud.shape[2] == 6:
              # These are written as bgr because rgb is interpreted as a single
              # little-endian float.
              dt = np.dtype([('x', np.float32),
                             ('y', np.float32),
                             ('z', np.float32),
                             ('b', np.uint8),
                             ('g', np.uint8),
                             ('r', np.uint8),
                             ('I', np.uint8)])
              pointCloud_tmp = np.zeros((height*width, 1), dtype=dt)
              for i, k in enumerate(['x', 'y', 'z', 'r', 'g', 'b']):
                  pointCloud_tmp[k] = pointCloud[:, :, i].reshape((height*width, 1))
              pointCloud_tmp.tofile(f)
          else:
              dt = np.dtype([('x', np.float32),
                             ('y', np.float32),
                             ('z', np.float32),
                             ('I', np.uint8)])
              pointCloud_tmp = np.zeros((height*width, 1), dtype=dt)
              for i, k in enumerate(['x', 'y', 'z']):
                  pointCloud_tmp[k] = pointCloud[:, :, i].reshape((height*width, 1))
              pointCloud_tmp.tofile(f)

def getRGBFromDepthTransform(calibration, camera, referenceCamera):
    irKey = "H_{0}_ir_from_{1}".format(camera, referenceCamera)
    rgbKey = "H_{0}_from_{1}".format(camera, referenceCamera)

    rgbFromRef = calibration[rgbKey][:]
    irFromRef = calibration[irKey][:]

    return np.dot(rgbFromRef, np.linalg.inv(irFromRef))


if __name__ == "__main__":
    reference_camera = "NP5"

    for target_object in object_list:
        print(target_object)
        if not os.path.exists(ycb_data_folder+"/"+target_object+"/clouds"):
    	    os.makedirs(ycb_data_folder+"/"+target_object+"/clouds")
        for viewpoint_camera in viewpoint_cameras:
            pbar = tqdm.tqdm(viewpoint_angles)
            for viewpoint_angle in pbar:
                pbar.set_description('Camera {0}, Angle {1}'.format(viewpoint_camera, viewpoint_angle))
                basename = "{0}_{1}".format(viewpoint_camera, viewpoint_angle)
                depthFilename = os.path.join(ycb_data_folder+ target_object, basename + ".h5")
                rgbFilename = os.path.join(ycb_data_folder + target_object, basename + ".jpg")
                maskFilename = os.path.join(ycb_data_folder + target_object + '/masks', basename + "_mask.pbm")

                calibrationFilename = os.path.join(ycb_data_folder+target_object, "calibration.h5")
                calibration = h5.File(calibrationFilename)

                if not os.path.isfile(rgbFilename):
                    print("The rgbd data is not available for the target object \"%s\". Please download the data first." % target_object)
                    exit(1)
                rgbImage = imread(rgbFilename)
                mask = imread(maskFilename)
                depthK = calibration["{0}_depth_K".format(viewpoint_camera)][:]
                rgbK = calibration["{0}_rgb_K".format(viewpoint_camera)][:]
                depthScale = np.array(calibration["{0}_ir_depth_scale".format(viewpoint_camera)]) * .0001 # 100um to meters
                H_RGBFromDepth = getRGBFromDepthTransform(calibration, viewpoint_camera, reference_camera)

                unregisteredDepthMap = h5.File(depthFilename)["depth"][:]
                unregisteredDepthMap = filterDiscontinuities(unregisteredDepthMap) * depthScale

                registeredDepthMap = registerDepthMap(unregisteredDepthMap,
                                                    rgbImage,
                                                    depthK,
                                                    rgbK,
                                                    H_RGBFromDepth)

                pointCloud = registeredDepthMapToPointCloud(registeredDepthMap, rgbImage, rgbK, organized=False, mask=mask)

                # writePLY(ycb_data_folder+target_object+"/clouds/pc_"+viewpoint_camera+"_"+referenceCamera+"_"+viewpoint_angle+".ply", pointCloud)
                writePCD(pointCloud, ycb_data_folder+target_object+"/clouds/"+viewpoint_camera+"_"+viewpoint_angle+".pcd", ascii=False)
