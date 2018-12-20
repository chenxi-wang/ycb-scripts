import os
import numpy as np
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

# object_list = ['063-a_marbles']

voxel_size = 0.002
dist_thresh = 0.005

def compute_distance(A,B):
    A = A[:,:3].copy()
    B = B[:,:3].copy().T
    A2 = np.sum(np.square(A), axis=1, keepdims=True)
    B2 = np.sum(np.square(B), axis=0, keepdims=True)
    AB = np.matmul(A, B)
    dists = A2 + B2 - 2 * AB
    dists = np.sqrt(dists + 1e-6)
    return dists


if __name__ == "__main__":
    reference_camera = "NP5"

    for target_object in object_list:
        print(target_object)
        output_dir = os.path.join(ycb_data_folder, target_object, "clouds_filtered")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model_file = os.path.join(ycb_data_folder, target_object, 'clouds', 'merged_cloud.ply')
        model = o3.read_point_cloud(model_file)
        sampled_model = o3.voxel_down_sample(model, voxel_size=voxel_size)
        sampled_model_points = np.asarray(sampled_model.points)

        for viewpoint_camera in viewpoint_cameras:
            pbar = tqdm.tqdm(viewpoint_angles)
            for viewpoint_angle in pbar:
                pbar.set_description('Camera {0}, Angle {1}'.format(viewpoint_camera, viewpoint_angle))
                
                cloud_file = os.path.join(ycb_data_folder, target_object, 'clouds_aligned', viewpoint_camera+'_'+viewpoint_angle+'.pcd')
                output_file = os.path.join(output_dir, viewpoint_camera+'_'+viewpoint_angle+'.pcd')
                # os.system('cp {0} {1}'.format(cloud_file, output_dir))
                os.system('cp empty_cloud.pcd {}'.format(output_file))

                cloud = o3.read_point_cloud(cloud_file)
                cloud_points = np.asarray(cloud.points)
                cloud_colors = np.asarray(cloud.colors)

                dists = compute_distance(cloud_points, sampled_model_points)
                mask = (dists.min(axis=1) < dist_thresh)

                filtered_cloud = o3.PointCloud()
                filtered_cloud.points = o3.Vector3dVector(cloud_points[mask])
                filtered_cloud.colors = o3.Vector3dVector(cloud_colors[mask])

                o3.write_point_cloud(output_file, filtered_cloud)
