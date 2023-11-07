import json
from pathlib import Path
from ...builder import DATASETS
from .topdown_coco_dataset import TopDownCocoDataset

# 박스 리사이즈
def resize_box(box, scale=1.2):
    x,y,w,h = box[0], box[1], box[2], box[3]
    x_mid = x + w/2
    y_mid = y + h/2
    w_new = w*scale
    h_new = h*scale

    x_new = x_mid - w_new/2
    y_new = y_mid - h_new/2

    return [x_new, y_new, w_new, h_new]

@DATASETS.register_module()
class TopDownNiaDataset(TopDownCocoDataset):
    def __init__(self,
                ann_dir,
                img_prefix,
                data_cfg,
                pipeline,
                dataset_info=None,
                test_mode=False):
        
        ann_dir_ = Path(ann_dir)
        print(ann_dir)
        ann_file = ann_dir_.parent / 'merged_annotations.json'


        if not (ann_file).is_file():
            print('[DATA PROCESSING] Merging annotation files...')
            annos = list(ann_dir_.glob('*.json'))
            merged_annos = dict()

            merged_annos['categories'] = [{
                'supercategory': 'person', 
                'id': 1, 'name': 'person',
                'keypoints': ['nose', \
                                'left_eye', \
                                'right_eye', \
                                'left_ear', \
                                'right_ear', \
                                'left_shoulder', \
                                'right_shoulder', \
                                'left_elbow', \
                                'right_elbow', \
                                'left_wrist', \
                                'right_wrist', \
                                'left_hip', \
                                'right_hip', \
                                'left_knee', \
                                'right_knee', \
                                'left_ankle', \
                                'right_ankle'], \
                'skeleton': [[16, 14], \
                            [14, 12], \
                            [17, 15], \
                            [15, 13], \
                            [12, 13], \
                            [6, 12], \
                            [7, 13], \
                            [6, 7], \
                            [6, 8], \
                            [7, 9], \
                            [8, 10], \
                            [9, 11], \
                            [2, 3], \
                            [1, 2], \
                            [1, 3], \
                            [2, 4], \
                            [3, 5], \
                            [4, 6], \
                            [5, 7]]}]
            
            merged_annos['images'] = list()
            merged_annos['annotations'] = list()

            for anno in ann_dir_.glob('*.json'):
                with anno.open() as f:
                    anno_json = json.load(f)
                merged_annos['images'].extend(anno_json['images'])
                merged_annos['annotations'].extend(anno_json['annotations'])
            
            # file_name에서 prefix 없에주기
            for idx, item in enumerate(merged_annos['images']):
                merged_annos['images'][idx]['file_name'] = item['file_name'].split('/')[-1]

            # annotations 에서 category_id 1로 바꿔주기
            # + box resize
            for idx, item in enumerate(merged_annos['annotations']):
                merged_annos['annotations'][idx]['category_id'] = 1
                merged_annos['annotations'][idx]['bbox'] = resize_box(item['bbox'], 1.2)

            with ann_file.open('w') as f:
                json.dump(merged_annos, f)
        
        # merged_annotations.json exists now
        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)
