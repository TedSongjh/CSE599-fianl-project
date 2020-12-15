
import os
from cruw.cruw import CRUW

import json
from cruw.eval.loc3d_cam.config_parser import  annotation_dict, load_loc3d_cam_config_dict
from cruw.eval.parser.cam_loader import CamResLoader


anno_config_name = 'anno_config'


anno_res_loader = annotation_dict(anno_config_name)
data_root = anno_res_loader.res_root
output_dir = anno_res_loader.output_dir

# load result
cruw = CRUW(data_root)
parser_config_name = anno_res_loader.parser_cfg
parser_cfg = load_loc3d_cam_config_dict(parser_config_name)
cam_res_loader = CamResLoader(data_root, cruw, parser_config_name)
cam_res_dict, gt_dict = cam_res_loader.load(verbose=True)

sensor_config = cruw.sensor_cfg
object_config = cruw.object_cfg

info = anno_res_loader.serialize()

"""reassemble data"""
for seq_name in anno_res_loader.seq_names:
    cam_valid = True
    rad_valid = True
    if cam_res_dict[seq_name] == None:
        cam_valid = False
    if gt_dict[seq_name] == None:
        rad_valid = False
    n_frame = 0
    seq_anno = {
        "seq_name": seq_name,
        "date_collect": "_".join(seq_name.split("_")[0:3]),
        "n_frame": n_frame
    }
    if cam_valid:
        n_frame = len(cam_res_dict[seq_name])
    elif rad_valid:
        n_frame = len(gt_dict[seq_name])
    else:
        seq_anno["valid"] = []
        if not cam_valid:
            seq_anno["valid"].append("missing camera annotation")
        if not rad_valid:
            seq_anno["valid"].append("missing radar annotation")


    metadata = []
    i = 0
    sensors = list(anno_res_loader.sensors)
    if cam_valid or rad_valid:
        for i in range(0,n_frame):
            frame_data = {
                "frame_id": i,
                "frame_name": ("000000%04d.jpg") % i,
            }
            if cam_valid:
                n = 0
                while n < int(sensors[1]):
                    n_object = len(cam_res_dict[seq_name][i])
                    frame_data["cam_%d" % n] = {
                        "folder_name": sensor_config.camera_cfg["image_folder"],
                        "width": sensor_config.camera_cfg["image_width"],
                        "height": sensor_config.camera_cfg["image_height"],
                        "n_object": len(cam_res_dict[seq_name][i]),

                    }
                    obj = 0
                    frame = cam_res_dict[seq_name][i]
                    while obj < n_object:
                        frame_data["cam_%d" % n]["obj_info_%d" % (obj+1)] = {
                            "categories": frame[obj]["type"],
                            "bboxes": frame[obj]["bbox"],
                            "visibilities": frame[obj]["visi"],
                            "truncations": frame[obj]["trunc"],
                            "source": frame[obj]["source"]
                        }
                        if "trans" in frame[obj]:
                            frame_data["cam_%d" % n]["obj_info_%d" % (obj+1)]["translation"] = frame[obj]["trans"]
                        obj += 1
                    n += 1

            if rad_valid and i < len(gt_dict[seq_name]):
                gt_seq = gt_dict[seq_name]
                gt = gt_seq[i]
                n = 0
                while n < int(sensors[3]):
                    rad_name = "radar_h"
                    if n == 1:
                        rad_name = "radar_v"
                    frame_data[rad_name] = {
                        "folder_name": sensor_config.radar_cfg["chirp_folder"],
                        "range": sensor_config.radar_cfg["ramap_rsize"],
                        "azimuth": sensor_config.radar_cfg["ramap_asize"],
                        "n_chirps": sensor_config.radar_cfg["n_chirps"],
                        "n_object": len(gt)

                    }
                    obj = 0
                    while obj < len(gt):
                        frame_data[rad_name]["obj_info_%d" % (obj+1)] = {
                            "categories": gt[obj]["type"],
                            "centers": gt[obj]["ra"],
                            "source": gt[obj]["source"]
                        }
                        obj += 1
                    n += 1
            metadata.append(frame_data)
        data = cam_res_dict[seq_name].copy()

    seq_anno.update(info)
    seq_anno["metadata"] = metadata


    #write json file
    output = os.path.join(output_dir, "%s.json" % seq_name)
    """
        if rad_valid and cam_valid:
        output=os.path.join(output_dir, "%s.json" % seq_name)
    elif cam_valid:
        output=os.path.join(output_dir, "%s_rad_invalid.json" % seq_name)
    elif rad_valid:
        output = os.path.join(output_dir, "%s_cam_invalid.json" % seq_name)
    else:
        output = os.path.join(output_dir, "%s_rad&cam_invalid.json" % seq_name)

    """

    with open(output, "w") as write_file:
        json.dump(seq_anno, write_file, indent=5)

