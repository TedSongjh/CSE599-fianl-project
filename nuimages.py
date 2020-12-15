import numpy as np
from tqdm import tqdm
from nuimages.nuimages import NuImages
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import  BoxVisibility
from pycocotools import mask

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.utils.file_io import PathManager

from .. import DatasetCatalog, MetadataCatalog

categories = ['human.pedestrian',
              'vehicle.car',
              'vehicle.bus',
              'vehicle.truck',
              'vehicle.cycle',
              'vehicle.cycle.withrider']
full_categories = ["animal",
                   "flat.driveable_surface",
                   "human.pedestrian.adult",
                   "human.pedestrian.child",
                   "human.pedestrian.construction_worker",
                   "human.pedestrian.personal_mobility",
                   "human.pedestrian.police_officer",
                   "human.pedestrian.stroller",
                   "human.pedestrian.wheelchair",
                   "movable_object.barrier",
                   "movable_object.debris",
                   "movable_object.pushable_pullable",
                   "movable_object.trafficcone",
                   "static_object.bicycle_rack",
                   "vehicle.bicycle",
                   "vehicle.bus.bendy",
                   "vehicle.bus.rigid",
                   "vehicle.car",
                   "vehicle.construction",
                   "vehicle.ego",
                   "vehicle.emergency.ambulance",
                   "vehicle.emergency.police",
                   "vehicle.motorcycle",
                   "vehicle.trailer",
                   "vehicle.truck"]
categories_mapping = [[2,3,4,5,6,7,8],
                      [17,18,20,21],
                      [15,16],
                      [24,25],
                      [14,22]]

def convert_categories(cid,categories_mapping):
    for i in range(len(categories_mapping)):
        if cid in categories_mapping[i]:
            return i
    return None

def load_nuimages_dicts(path, version, categories = categories):
    assert (path[-1] == "/"), "Insert '/' in the end of path"
    nuim = NuImages(dataroot='/mnt/disk1/nuImages', version=version, verbose=True, lazy=True)

    if categories == None:
        categories = [data["name"] for data in nuim.category]
    assert (isinstance(categories, list)), "Categories type must be list"

    dataset_dicts = []
    #idx = 0
    # for i in tqdm(range(0, len(nuim.scene))):
        # scene = nuim.scene[i]
        # scene_rec = nuim.get('scene', scene['token'])
        # sample_rec_cur = nuim.get('sample', scene_rec['first_sample_token'])

        # Go through all frame in current scene
    flag = 1
    for idx in tqdm(range(0, len(nuim.sample))):
        data = nuim.sample_data[idx]
        # if not nuim.get('calibrated_sensor', data['calibrated_sensor_token'])['sensor_token']=="23b8c1e9392446debeb13b9046685257":
        # #if not data['filename'][6:17] =="/CAM_FRONT/":
        #     continue
        if not (data['filename'][:17] =="sweeps/CAM_FRONT/" or data['filename'][:18] =="samples/CAM_FRONT/"):
            continue
        record = {}
        record["file_name"] = path + data["filename"]
        record["image_id"] = idx
        record["height"] = data["height"]
        record["width"] = data["width"]
        #idx += 1

        # Get sample_content
        objs = []
        if data['is_key_frame']:
            #print(version,data['filename'][:18])

            #content = nuim.get_sample_content(data['sample_token'])
            nuim.load_tables(['object_ann','sample_data','category','attribute'])
            #print(nuim.object_ann[0])
            objects = []
            for i in nuim.object_ann:
                if i['sample_data_token']==nuim.sample_data[idx]['token']:
                    objects.append(i)

            #print(boxes)

            #boxes = [[11,12,13,14]]
            _, segs = nuim.get_segmentation(data['token'])
            objnum=1
            for object in objects:

                #seg = np.zeros(segs.shape)
                seg = (segs == objnum)
                seg = seg.astype('uint8')
                #seg = segs[selection]
                # for x in range(len(segs)):
                #     for y in range(len(segs[0])):
                #         if segs[x][y] == objnum:
                #             seg[x][y] = 1


                for j in range(len(nuim.category)):
                    if nuim.category[j]['token'] == object['category_token']:
                        catid = j
                        break
                catid = convert_categories(catid,categories_mapping)
                if catid == None:
                    continue
                if catid == 4:
                    if object['attribute_tokens']== nuim.attribute[0]['token']:
                        catid = 5

                obj = {
                    "bbox": object['bbox'],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": catid,
                    "iscrowd": 0,
                    "segmentation": mask.encode(np.asarray(seg, order="F"))
                }
                objs.append(obj)
                objnum += 1

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

root_path = '/mnt/disk1/nuImages/'


# categories = ['human.pedestrian.adult',
#               'human.pedestrian.child',
#               'human.pedestrian.stroller',
#               'human.pedestrian.personal_mobility',
#               'human.pedestrian.police_officer',
#               'human.pedestrian.construction_worker',
#               'vehicle.car',
#               'vehicle.bus.bendy',
#               'vehicle.bus.rigid',
#               'vehicle.truck',
#               'vehicle.trailer']
#
# dataset = 'nuimages_test'
# version = 'v1.0-test'
#
# get_dicts = lambda p = root_path, c = categories: load_nuimages_dicts(path=p,version = version, categories=c)
# DatasetCatalog.register(dataset,get_dicts)
# MetadataCatalog.get(dataset).thing_classes = categories
# MetadataCatalog.get(dataset).evaluator_type = "coco"
#
# dataset = 'nuimages_train'
# version = 'v1.0-train'
# get_dicts = lambda p = root_path, c = categories: load_nuimages_dicts(path=p,version = version, categories=c)
# DatasetCatalog.register(dataset,get_dicts)
# MetadataCatalog.get(dataset).thing_classes = categories
# MetadataCatalog.get(dataset).evaluator_type = "coco"
#



# dataset = 'nuimages_mini'
# version = 'v1.0-mini'
#
# get_dicts = lambda p = root_path, c = categories: load_nuimages_dicts(path=p,version = version, categories=c)
# DatasetCatalog.register(dataset,get_dicts)
# MetadataCatalog.get(dataset).thing_classes = categories
# MetadataCatalog.get(dataset).evaluator_type = "coco"