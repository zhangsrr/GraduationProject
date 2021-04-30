"""
test streamlines_segment
"""
from backend.streamlines_segment_tool import *
import sys
import backend.global_define as glo
from datetime import datetime

if __name__ == '__main__':
    vtk_name = 'tec025.5044_00_type00_streamlines_40_both_reduced_40'
    # path = "/home/gp2021-zsr/gp/"
    path = "C:/AllSoftwareDisk/PycharmProjects/GraduationProject/"
    src_streamlines_filename = path + "data/" + vtk_name + ".vtk"
    dest_streamlines_filename_prefix = path + "data/" + vtk_name + "_processed_"
    dest_streamlines_count_list = [1, 2, 4, 8, 16, 32, 40]

    glo._init()
    glo.set_value('starttime', datetime.now())
    glo.set_value('endtime', datetime.now())
    print("Time: " + str(glo.time_pass()) + '\n')

    print(">>>>>>>>>>>>>>>[" + vtk_name + ".vtk]<<<<<<<<<<<<<")
    print("Start filtering streamlines...")
    obj = StreamlinesSegment()
    # Input arguments below
    # python test.py MeanShift
    # cluster_mode = "MeanShift"
    if len(sys.argv) != 2:
        print("Error arguments input, should be python test.py cluster_mode")
        print("e.g. python test.py KMeans")
        print("Do you want to run by KMeans default?")
        ans = input("yes or no:\n")
        assert (ans == "yes" or ans == "no")
        if ans == "yes":
            cluster_mode = "KMeans"
        else:
            exit(-1)
    else:
        cluster_mode = sys.argv[1]
    assert (cluster_mode == "KMeans" or
            cluster_mode == "KMedoids" or
            cluster_mode == "DBSCAN" or
            cluster_mode == "OPTICS" or
            cluster_mode == "MeanShift" or
            cluster_mode == "AP" or
            cluster_mode == "AffinityPropagation" or
            cluster_mode == "CURE")
    obj.filter_streamlines(src_streamlines_filename,
                           dest_streamlines_count_list,
                           dest_streamlines_filename_prefix,
                           cluster_mode)
    print("Running finished.")
    glo.set_value('endtime', datetime.now())
    print("Time: " + str(glo.time_pass()) + '\n')

print("Done.")
