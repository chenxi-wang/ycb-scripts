import os
import sys
import json
import urllib
import urllib2

output_directory = "./ycb"

# You can either set this to "all" or a list of the objects that you'd like to
# download.
# objects_to_download = "all"
#objects_to_download = ["002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can"]
# objects_to_download = ["006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", "011_banana", "012_strawberry", "013_apple", "014_lemon", "015_peach", "016_pear", "017_orange", "018_plum", "019_pitcher_base", "021_bleach_cleanser", "022_windex_bottle", "024_bowl", "026_sponge", "027_skillet", "029_plate", "030_fork", "031_spoon", "032_knife", "033_spatula", "035_power_drill", "036_wood_block", "037_scissors", "038_padlock", "039_key", "040_large_marker", "041_small_marker", "042_adjustable_wrench", "043_phillips_screwdriver", "044_flat_screwdriver", "048_hammer", "049_small_clamp", "050_medium_clamp", "051_large_clamp", "052_extra_large_clamp", "053_mini_soccer_ball", "054_softball", "055_baseball", "056_tennis_ball", "057_racquetball", "058_golf_ball", "059_chain", "061_foam_brick", "062_dice", "063-a_marbles", "065-a_cups", "065-b_cups", "065-c_cups", "065-d_cups", "065-e_cups", "065-f_cups", "065-g_cups", "065-h_cups", "065-i_cups", "065-j_cups", "070-a_colored_wood_blocks", "071_nine_hole_peg_test", "072-a_toy_airplane", "072-b_toy_airplane", "072-c_toy_airplane", "072-d_toy_airplane", "072-e_toy_airplane", "072-f_toy_airplane", "072-g_toy_airplane", "072-h_toy_airplane", "072-i_toy_airplane", "072-j_toy_airplane", "072-k_toy_airplane", "073-a_lego_duplo", "073-b_lego_duplo", "073-c_lego_duplo", "073-d_lego_duplo", "073-e_lego_duplo", "073-f_lego_duplo", "073-g_lego_duplo", "073-h_lego_duplo", "073-i_lego_duplo", "073-j_lego_duplo", "073-k_lego_duplo", "073-l_lego_duplo", "073-m_lego_duplo", "076_timer", "077_rubiks_cube"]
# objects_to_download = ["025_mug"]

f = open(os.path.join(output_directory, 'objects.txt'))
objects_to_download = f.readlines()
f.close()

# You can edit this list to only download certain kinds of files.
# 'berkeley_rgbd' contains all of the depth maps and images from the Carmines.
# 'berkeley_rgb_highres' contains all of the high-res images from the Canon cameras.
# 'berkeley_processed' contains all of the segmented point clouds and textured meshes.
# 'google_16k' contains google meshes with 16k vertices.
# 'google_64k' contains google meshes with 64k vertices.
# 'google_512k' contains google meshes with 512k vertices.
# See the website for more details.
#files_to_download = ["berkeley_rgbd", "berkeley_rgb_highres", "berkeley_processed", "google_16k", "google_64k", "google_512k"]
files_to_download = ["berkeley_rgbd", "berkeley_rgb_highres" "berkeley_processed"]

# Extract all files from the downloaded .tgz, and remove .tgz files.
# If false, will just download all .tgz files to output_directory
extract = True

base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
objects_url = base_url + "objects.json"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def fetch_objects(url):
    response = urllib2.urlopen(url)
    html = response.read()
    objects = json.loads(html)
    return objects["objects"]

def download_file(url, filename):
    u = urllib2.urlopen(url)
    f = open(filename, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print("Downloading: %s (%s MB)" % (filename, file_size/1000000.0))

    file_size_dl = 0
    block_sz = 65536
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl/1000000.0, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print(status,)
    f.close()

def tgz_url(object, type):
    if type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return base_url + "berkeley/{object}/{object}_{type}.tgz".format(object=object,type=type)
    elif type in ["berkeley_processed"]:
        return base_url + "berkeley/{object}/{object}_berkeley_meshes.tgz".format(object=object,type=type)
    else:
        return base_url + "google/{object}_{type}.tgz".format(object=object,type=type)

def extract_tgz(filename, dir):
    tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename,dir=dir)
    os.system(tar_command)
    os.remove(filename)

def check_url(url):
    try:
        request = urllib2.Request(url)
        request.get_method = lambda : 'HEAD'
        response = urllib2.urlopen(request)
        return True
    except Exception as e:
        return False


if __name__ == "__main__":

    objects = objects_to_download
    # objects = fetch_objects(objects_url)

    for object in objects:
        if objects_to_download == "all" or object in objects_to_download:
            for file_type in files_to_download:
                url = tgz_url(object, file_type)
                if not check_url(url):
                    continue
                filename = "{path}/{object}_{file_type}.tgz".format(path=output_directory,
                                                                    object=object,
                                                                    file_type=file_type)
                download_file(url, filename)
                if extract:
                    extract_tgz(filename, output_directory)
