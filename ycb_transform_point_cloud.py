import os
import numpy as np
import h5py as h5
import math
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

# object_list = ['072-a_toy_airplane']

def getTransform(calibration, viewpoint_camera, referenceCamera, transformation):
    CamFromRefKey = "H_{0}_from_{1}".format(viewpoint_camera, referenceCamera)
    CamFromRef = calibration[CamFromRefKey][:]

    TableFromRefKey = "H_table_from_reference_camera"
    TableFromRef = transformation[TableFromRefKey][:]
    return np.dot(TableFromRef, np.linalg.inv(CamFromRef))


if __name__ == "__main__":
    reference_camera = "NP5"

    for target_object in object_list:
        print(target_object)
        output_dir = os.path.join(ycb_data_folder, target_object, "clouds_aligned")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for viewpoint_camera in viewpoint_cameras:
            pbar = tqdm.tqdm(viewpoint_angles)
            for viewpoint_angle in pbar:
                pbar.set_description('Camera {0}, Angle {1}'.format(viewpoint_camera, viewpoint_angle))
                
                cloud_file = os.path.join(ycb_data_folder, target_object, 'clouds', viewpoint_camera+'_'+viewpoint_angle+'.pcd')
                output_file = os.path.join(output_dir, viewpoint_camera+'_'+viewpoint_angle+'.pcd')
                # os.system('cp {0} {1}'.format(cloud_file, output_dir))
                os.system('cp empty_cloud.pcd {}'.format(output_file))

                calibrationFilename = os.path.join(ycb_data_folder+target_object, "calibration.h5")
                transformationFilename = os.path.join(ycb_data_folder, target_object, 'poses', '{}_{}_pose.h5'.format(reference_camera, viewpoint_angle))
                
                calibration = h5.File(calibrationFilename)
                transformation = h5.File(transformationFilename)
                H_table_from_cam = getTransform(calibration, viewpoint_camera, reference_camera, transformation)

                pointCloud = o3.read_point_cloud(cloud_file)
                points = np.asarray(pointCloud.points)
                ones = np.ones(len(points))[:,np.newaxis]
                
                points_ = np.concatenate([points,ones], axis=1)
                aligned_points_ = np.matmul(H_table_from_cam, points_.T)
                aligned_points = aligned_points_.T[:,:3]

                aligned_cloud = o3.PointCloud()
                aligned_cloud.points = o3.Vector3dVector(aligned_points)
                aligned_cloud.colors = pointCloud.colors

                o3.write_point_cloud(output_file, aligned_cloud)
