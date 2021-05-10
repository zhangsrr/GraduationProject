"""
test streamlines_segment
"""
from backend.streamlines_segment_tool import *
from backend.helper_function import output_time
import sys


if __name__ == '__main__':
    # vtk_name = 'tec026.0031_00_type00_streamlines_40_both_reduced_40'
    # vtk_name = 'tec026.0031_00_type00_streamlines_40_both_reduced_200'
    # vtk_name = 'tec025.5044_00_type00_streamlines_40_both'
    vtk_name = 'tec026.0031_00_type00_streamlines_40_both'
    # path = "/home/gp2021-zsr/gp/"
    path = "C:/AllSoftwareDisk/PycharmProjects/GraduationProject/"
    src_streamlines_filename = path + "data/" + vtk_name + ".vtk"
    dest_streamlines_filename_prefix = path + "data/" + vtk_name + "_processed_"
    # dest_streamlines_count_list = [1, 2, 4, 8, 16, 32, 40]

    dest_streamlines_count_list = [16]

    print("Time: " + output_time() + '\n')

    print(">>>>>>>>>>>>>>>[" + vtk_name + ".vtk]<<<<<<<<<<<<<<<")
    print("Start filtering streamlines...")
    obj = StreamlinesSegment()
    print(">>>>>>>>>>>>>>> choose clustering algorithm <<<<<<<<<<<<<<<")
    print("0. All")
    print("1. KMeans")
    print("2. KMedoids")
    print("3. DBSCAN")
    print("4. OPTICS")
    print("5. MeanShift")
    print("6. Affinity Propagation")
    print("7. CURE")
    cluster_idx = input("clustering algorithm index: ")
    if cluster_idx == '0':
        cluster_mode = 'All'
    elif cluster_idx == '1':
        cluster_mode = "KMeans"
    elif cluster_idx == '2':
        cluster_mode = "KMedoids"
    elif cluster_idx == '3':
        cluster_mode = "DBSCAN"
    elif cluster_idx == '4':
        cluster_mode = "OPTICS"
    elif cluster_idx == '5':
        cluster_mode = "MeanShift"
    elif cluster_idx == '6':
        cluster_mode = "AffinityPropagation"
    elif cluster_idx == '7':
        cluster_mode = "CURE"
    else:
        cluster_mode = None
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # print(cluster_mode)
    # Input arguments below
    # python test.py MeanShift
    # cluster_mode = "MeanShift"
    # if len(sys.argv) != 2:
    #     print("Error arguments input, should be python test.py cluster_mode")
    #     print("e.g. python test.py KMeans")
    #     print("Do you want to run by KMeans default?")
    #     ans = input("yes or no:\n")
    #     assert (ans == "yes" or ans == "no")
    #     if ans == "yes":
    #         cluster_mode = "KMeans"
    #     else:
    #         exit(-1)
    # else:
    #     cluster_mode = sys.argv[1]
    assert (cluster_mode == "KMeans" or
            cluster_mode == "KMedoids" or
            cluster_mode == "DBSCAN" or
            cluster_mode == "OPTICS" or
            cluster_mode == "MeanShift" or
            cluster_mode == "AP" or
            cluster_mode == "AffinityPropagation" or
            cluster_mode == "CURE" or
            cluster_mode == "All")

    obj.filter_streamlines(src_streamlines_filename,
                           dest_streamlines_count_list,
                           dest_streamlines_filename_prefix,
                           cluster_mode)
    print("Running finished.")
    print("Time: " + output_time() + '\n')

print("Done.")
