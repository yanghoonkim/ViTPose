from pathlib import Path
import pandas as pd
import json

from nia.nia_dataset_reader import (
    NiaDataPathExtractor,
    DataFrameSplitter,
    NiaKeypointDataPathProvider,
)
               

categories = [{'supercategory': 'person',
  'id': 1,
  'name': 'person',
  'keypoints': ['nose',
   'left_eye',
   'right_eye',
   'left_ear',
   'right_ear',
   'left_shoulder',
   'right_shoulder',
   'left_elbow',
   'right_elbow',
   'left_wrist',
   'right_wrist',
   'left_hip',
   'right_hip',
   'left_knee',
   'right_knee',
   'left_ankle',
   'right_ankle'],
  'skeleton': [[16, 14],
   [14, 12],
   [17, 15],
   [15, 13],
   [12, 13],
   [6, 12],
   [7, 13],
   [6, 7],
   [6, 8],
   [7, 9],
   [8, 10],
   [9, 11],
   [2, 3],
   [1, 2],
   [1, 3],
   [2, 4],
   [3, 5],
   [4, 6],
   [5, 7]]}]


BASE_PATH = Path('/root/ViTPose/data/nia/')
ANNO_PATH = BASE_PATH / '2.라벨링데이터'
COLL_PATH = BASE_PATH / '1.원천데이터'
TRAIN_LABEL_PATH = BASE_PATH / 'keypoint_train_label.json'
VALID_LABEL_PATH = BASE_PATH / 'keypoint_valid_label.json'
TEST_LABEL_PATH = BASE_PATH / 'keypoint_test_label.json'
VALID_BOX_PATH = BASE_PATH / 'valid_boxes.json'
TEST_BOX_PATH = BASE_PATH / 'test_boxes.json'

def to_frame(pairs):
    df = pd.DataFrame(pairs, columns=['imgpath', 'annopath'])
    df.index = df.imgpath.apply(lambda x: x.split('/')[-1])
    df.index.name = 'filename'
    return df


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


def make_dict(df):
    anno_images = list()
    anno_annotations = list()

    for filename, item in zip(df.imgpath, df.annopath):
        with open(item) as f:
            item_json = json.load(f)

        anno_images.extend(item_json['images'])
        anno_images[-1]['file_name'] = Path(filename).relative_to('/root/ViTPose/data/nia/1.원천데이터/').as_posix()
        anno_annotations.extend(item_json['annotations'])

    for idx, item in enumerate(anno_annotations):
        anno_annotations[idx]['bbox'] = resize_box(item['bbox'], 1.2)
    
    dict_ = {'categories': categories, 'images': anno_images, 'annotations': anno_annotations}

    return dict_


def split_data():
    if (not TRAIN_LABEL_PATH.exists()) or (not VALID_LABEL_PATH.exists()) or (not TEST_LABEL_PATH.exists()):
        print('[DATA SPLIT] Splitting data...')

        path_provider = NiaKeypointDataPathProvider(
            visible_reader=NiaDataPathExtractor(
                dataset_dir=BASE_PATH.as_posix(),
                pattern=(
                    r"(?P<type>[^/]+)/"
                    r"(?P<collector>[^/]+)/"
                    r".*?"
                    r"(?P<channel>[^/]+)/"
                    r"(?P<filename>[^/]+)$"
                ),
            ),
            keypoint_reader=NiaDataPathExtractor(
                dataset_dir=BASE_PATH.as_posix(),
                pattern=(
                    r"(?P<type>[^/]+)/"
                    r"(?P<channel>[^/]+)/"
                    r"(?P<filename>[^/]+)$"
                ),
            ),
            splitter=DataFrameSplitter(
                groups=["channel", "collector", "scene", "road", "timeslot", "weather"],
                splits=["train", "valid", "test"],
                ratios=[8, 1, 1],
                seed=231111,
            ),
            channels=["image_B", "image_F", "image_L", "image_R"],
        )

        train_path_pairs = path_provider.get_split_data_list("train")
        valid_path_pairs = path_provider.get_split_data_list('valid')
        test_path_pairs = path_provider.get_split_data_list('test')

        df_thermal_train = to_frame(train_path_pairs)
        df_thermal_valid = to_frame(valid_path_pairs)
        df_thermal_test = to_frame(test_path_pairs)

        train_dict = make_dict(df_thermal_train)
        valid_dict = make_dict(df_thermal_valid)
        test_dict = make_dict(df_thermal_test)

        '''
        # annotation id 중복 이슈 해결
        temp = set()
        anno_id = 0
        for idx, item in enumerate(train_dict['annotations']):
            train_dict['annotations'][idx]['id'] = anno_id
            anno_id += 1
        for idx, item in enumerate(valid_dict['annotations']):
            valid_dict['annotations'][idx]['id'] = anno_id
            anno_id += 1
        for idx, item in enumerate(test_dict['annotations']):
            test_dict['annotations'][idx]['id'] = anno_id
            anno_id += 1
        '''

        with TRAIN_LABEL_PATH.open('w') as f:
            json.dump(train_dict, f)
        
        with VALID_LABEL_PATH.open('w') as f:
            json.dump(valid_dict, f)

        with TEST_LABEL_PATH.open('w') as f:
            json.dump(test_dict, f)
        
        # person detection results
        valid_box_json = list()
        for idx, item in enumerate(valid_dict['annotations']):
            temp_dict = dict()
            temp_dict['bbox'] = item['bbox']
            temp_dict['category_id'] = 1
            temp_dict['image_id'] = item['image_id']
            temp_dict['score'] = 0.99
            valid_box_json.append(temp_dict)

        with open(VALID_BOX_PATH, 'w') as f:
            json.dump(valid_box_json, f)

        test_box_json = list()
        for idx, item in enumerate(test_dict['annotations']):
            temp_dict = dict()
            temp_dict['bbox'] = item['bbox']
            temp_dict['category_id'] = 1
            temp_dict['image_id'] = item['image_id']
            temp_dict['score'] = 0.99
            test_box_json.append(temp_dict)

        with open(TEST_BOX_PATH, 'w') as f:
            json.dump(test_box_json, f)

    else:
        print('[DATA SPLIT] Load existing files...')   

