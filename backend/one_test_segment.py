"""
test streamlines_segment
"""
from backend.streamlines_segment import *

if __name__ == '__main__':
    vtk_name = 'tec025.5044_00_type00_streamlines_40_both'
    path = "/home/gp2021-zsr/gp/"
    src_streamlines_filename = path + "data/" + vtk_name + ".vtk"
    dest_streamlines_filename_prefix = path + "data/" + vtk_name + "_processed_"
    dest_streamlines_count_list = [1, 2, 4, 8, 16, 32, 40]

    print(">>>>>>>>>>>>>>>[" + vtk_name + ".vtk]<<<<<<<<<<<<<")
    print("Start filtering streamlines...")
    obj = Segment()
    obj.filter_streamlines(src_streamlines_filename,
                           dest_streamlines_count_list,
                           dest_streamlines_filename_prefix)
    print("Running finished.")

print("Done.")
