import os
import six
import glob
import json
import numpy as np
from collections import OrderedDict
from PIL import Image
import shutil
import tqdm

#from toolkit.datasets import ioutils
from pycocotools.mask import encode



class EgoExo4D(object):
 
    def __init__(self, frames_dir, anno_dir, download=True, fps=30, resolution=720, mode='lt', seq_names=None):
        super(EgoExo4D, self).__init__()

        self.frames_dir = frames_dir
        self.anno_dir = anno_dir
        self.fps = fps
        self.resolution = resolution
        self.mode = mode

        #self.seq_files = sorted(os.listdir(self.anno_files_path))#[-300:-297]
        #self.seq_names = [sf.split('.')[0] for sf in self.seq_files]

        resol_str = f"{self.resolution}" if self.resolution > 0 else ""
        #if self.fps == 30:
        #    resol_str += '/all'
        #self.seq_names = np.genfromtxt(os.path.join(anno_dir, self.mode, resol_str, 'sequences.txt'), delimiter='\n', dtype='str').tolist()#[:2]
        if 'val' in self.anno_dir:
            self.seq_names = np.sort(np.genfromtxt(os.path.join(anno_dir, self.mode, resol_str, 'sequences_v2.txt'), delimiter='\n', dtype='str').tolist())#[:400]
        else:
            self.seq_names = np.genfromtxt(os.path.join(anno_dir, self.mode, resol_str, 'sequences.txt'), delimiter='\n', dtype='str').tolist()#[:2]
        print(len(self.seq_names))
        #if self.fps == 30:
        #    self.seq_names.remove('upenn_0711_Cooking_6_2*onion_0*1')
        #    self.seq_names.remove('sfu_cooking_007_3*blue plate_0*19')
        #    self.seq_names.remove('sfu_cooking_007_1*olive oil bottle_0*1')

        if seq_names is not None:
            self.seq_names = seq_names


    def __getitem__(self, index):
        """
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files_fpv, img_files_tpv, anno_fpv, anno_tpv), where:
                - img_files_fpv: List of file names for frames with 'aria' from frames_fpv
                - img_files_tpv: List of file names for frames without 'aria' from frames_tpv
                - anno_fpv: NumPy array containing annotations for frames with 'aria'
                - anno_tpv: NumPy array containing annotations for frames without 'aria'
        """
        #seq_file = self.seq_files[index]
        #seq_name = seq_file.split('.')[0]
        #take_name = seq_name.split('*')[0]

        seq_name = self.seq_names[index]

        if self.mode == 'st':
            seq_name_parts = seq_name.split('$')
            st_sq_idx = seq_name_parts[-1]
            st_seq_name = seq_name_parts[0]


        take_name = seq_name.split('*')[0]
        resol_str = f"{self.resolution}" if self.resolution > 0 else ""
        #if self.fps == 30:
        #    resol_str += '/all'
        
        if self.mode == 'lt':
            keys = sorted(os.listdir(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos')))
        else:
            keys = sorted(os.listdir(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos')))
        #print(keys, seq_name)
        ego_key = keys[0]
        exo_key = keys[1]
        assert 'aria' in ego_key, keys

        if self.mode == 'lt':
            files_dir_ego = os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{ego_key}')
        else:
            files_dir_ego = os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{ego_key}', st_sq_idx)
        
        boxes_ego = np.genfromtxt(os.path.join(files_dir_ego, 'boxes.txt'), delimiter=',')
        frame_idxs_ego = np.genfromtxt(os.path.join(files_dir_ego, 'frames.txt'), delimiter='\n', dtype=np.int64)
        visibilities_ego = np.genfromtxt(os.path.join(files_dir_ego, 'visibilities.txt'), delimiter='\n')

        if self.mode == 'lt':
            files_dir_exo = os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{exo_key}')
        else:
            files_dir_exo = os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{exo_key}', st_sq_idx)

        boxes_exo = np.genfromtxt(os.path.join(files_dir_exo, 'boxes.txt'), delimiter=',')
        frame_idxs_exo = np.genfromtxt(os.path.join(files_dir_exo, 'frames.txt'), delimiter='\n', dtype=np.int64)
        visibilities_exo = np.genfromtxt(os.path.join(files_dir_exo, 'visibilities.txt'), delimiter='\n')

        assert frame_idxs_ego.shape[0] == frame_idxs_exo.shape[0]
        assert boxes_exo.shape == boxes_ego.shape

        """
        with open(os.path.join(self.anno_files_path, seq_file), "r") as f:
            annotations = json.load(f) 

        keys = sorted(list(annotations[seq_name].keys()))
        #print(keys, seq_name)
        ego_key = keys[0]
        exo_key = keys[1]
        assert 'aria' in ego_key, keys
        
        anno_frame_idxs_fpv = list(annotations[seq_name][ego_key].keys())
        anno_frame_idxs_fpv = [int(fi) for fi in anno_frame_idxs_fpv]
        boxes_fpv = [annotations[seq_name][ego_key][str(fi)]['box'] for fi in anno_frame_idxs_fpv]
        
        anno_frame_idxs_tpv = list(annotations[seq_name][exo_key].keys())
        anno_frame_idxs_tpv = [int(fi) for fi in anno_frame_idxs_tpv]
        boxes_tpv = [annotations[seq_name][exo_key][str(fi)]['box'] for fi in anno_frame_idxs_tpv]

        assert anno_frame_idxs_fpv == anno_frame_idxs_tpv
        assert len(boxes_fpv) == len(boxes_tpv)
        """

        if self.fps == 1:
            images_ego = [os.path.join(self.frames_dir, f'{self.resolution}', 'takes', take_name, 'frame_aligned_videos', f'{ego_key}', f'{fi}.jpg') for fi in frame_idxs_ego]
            images_exo = [os.path.join(self.frames_dir, f'{self.resolution}', 'takes', take_name, 'frame_aligned_videos', f'{exo_key}', f'{fi}.jpg') for fi in frame_idxs_exo]

            #boxes_fpv = np.array(boxes_fpv)
            #boxes_tpv = np.array(boxes_tpv)
        elif self.fps == 5:
            frame_idxs = list(range(frame_idxs_ego.min(), frame_idxs_ego.max()+1, 6))

            images_ego = [os.path.join(self.frames_dir, f'{self.resolution}', 'takes', take_name, 'frame_aligned_videos', f'{ego_key}', f'{fi}.jpg') for fi in frame_idxs]
            images_exo = [os.path.join(self.frames_dir, f'{self.resolution}', 'takes', take_name, 'frame_aligned_videos', f'{exo_key}', f'{fi}.jpg') for fi in frame_idxs]

            boxes_ego_ = np.zeros((len(images_ego), 4)) + np.nan
            boxes_exo_ = np.zeros((len(images_exo), 4)) + np.nan
            for i, fi in enumerate(frame_idxs_ego):
                i_ = frame_idxs.index(fi)
                boxes_ego_[i_] = boxes_ego[i]
                boxes_exo_[i_] = boxes_exo[i]

            boxes_ego = boxes_ego_
            boxes_exo = boxes_exo_
        elif self.fps == 10:
            frame_idxs = list(range(frame_idxs_ego.min(), frame_idxs_ego.max()+1, 3))

            images_ego = [os.path.join(self.frames_dir, f'{self.resolution}', 'takes', take_name, 'frame_aligned_videos', f'{ego_key}', f'{fi}.jpg') for fi in frame_idxs]
            images_exo = [os.path.join(self.frames_dir, f'{self.resolution}', 'takes', take_name, 'frame_aligned_videos', f'{exo_key}', f'{fi}.jpg') for fi in frame_idxs]

            boxes_ego_ = np.zeros((len(images_ego), 4)) + np.nan
            boxes_exo_ = np.zeros((len(images_exo), 4)) + np.nan
            for i, fi in enumerate(frame_idxs_ego):
                i_ = frame_idxs.index(fi)
                boxes_ego_[i_] = boxes_ego[i]
                boxes_exo_[i_] = boxes_exo[i]

            boxes_ego = boxes_ego_
            boxes_exo = boxes_exo_
        elif self.fps == 30:
            
            frame_idxs = list(range(frame_idxs_ego.min(), frame_idxs_ego.max()+1,30))
            print(frame_idxs)

            images_ego = [os.path.join(self.frames_dir, f'{self.resolution}', 'takes', take_name, 'frame_aligned_videos', f'{ego_key}', f'{fi}.jpg') for fi in frame_idxs]
            images_exo = [os.path.join(self.frames_dir, f'{self.resolution}', 'takes', take_name, 'frame_aligned_videos', f'{exo_key}', f'{fi}.jpg') for fi in frame_idxs]

            boxes_ego_ = np.zeros((len(images_ego), 4)) + np.nan
            boxes_exo_ = np.zeros((len(images_exo), 4)) + np.nan
            for i, fi in enumerate(frame_idxs_ego):
                i_ = frame_idxs.index(fi)
                boxes_ego_[i_] = boxes_ego[i]
                boxes_exo_[i_] = boxes_exo[i]

            boxes_ego = boxes_ego_
            boxes_exo = boxes_exo_


        return images_ego, images_exo, boxes_ego, boxes_exo
    

    def __len__(self):
        return len(self.seq_names)


    def export_dataset(self, split='test', copy_frames=False):

        SAVE_DIR = '/media/TBData2/VISTA'

        if copy_frames:
            SAVE_DIR = SAVE_DIR + '-w-frames'

        spl = 'val' if split == 'test' else 'train'

        MASKS_DIR = f'/media/TBData2/projects/EgoExo4d/v2/annotations/vot_ego_exo/vos/{spl}'


        ATTR_DIR = f'/media/TBData2/projects/EgoExo4d/v2/annotations/vot_ego_exo/sot/{spl}/lt/'
        ATTR_DIR = ATTR_DIR + f"{self.resolution}" if self.resolution > 0 else ATTR_DIR

        attributes = ['scale_variation', 'aspect_ratio_change', 'low_resolution', 'medium_resolution', 'high_resolution', 'fast_motion', 'motion_blur', 'distractors', 'illumination_variation',
        'object_static', 'object_moving', 'hoi']

        ALIGNED_TRACK_DIR = f'/media/TBData2/projects/EgoExo4d/v2/annotations/vot_ego_exo/aligned_tracks/{spl}'
        

        annotation_dict = {}

        for seq_name in tqdm.tqdm(self.seq_names):
            #seq_name = self.seq_names[index]

            if self.mode == 'st':
                seq_name_parts = seq_name.split('$')
                st_sq_idx = seq_name_parts[-1]
                st_seq_name = seq_name_parts[0]


            take_name = seq_name.split('*')[0]
            resol_str = f"{self.resolution}" if self.resolution > 0 else ""

            if self.mode == 'lt':
                with open(os.path.join(ALIGNED_TRACK_DIR, f'{seq_name}.json'), "r") as f:
                    refactored_anno = json.load(f) 
            else:
                with open(os.path.join(ALIGNED_TRACK_DIR, f'{st_seq_name}.json'), "r") as f:
                    refactored_anno = json.load(f) 

            #if self.fps == 30:
            #    resol_str += '/all'
            
            if self.mode == 'lt':
                keys = sorted(os.listdir(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos')))
            else:
                keys = sorted(os.listdir(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos')))
            #print(keys, seq_name)
            ego_key = keys[0]
            exo_key = keys[1]
            assert 'aria' in ego_key, keys

            if self.mode == 'lt':
                files_dir_ego = os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{ego_key}')
            else:
                files_dir_ego = os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{ego_key}', st_sq_idx)
            
            boxes_ego = np.genfromtxt(os.path.join(files_dir_ego, 'boxes.txt'), delimiter=',')
            frame_idxs_ego = np.genfromtxt(os.path.join(files_dir_ego, 'frames.txt'), delimiter='\n', dtype=np.int64)
            visibilities_ego = np.genfromtxt(os.path.join(files_dir_ego, 'visibilities.txt'), delimiter='\n')

            if self.mode == 'lt':
                files_dir_exo = os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{exo_key}')
            else:
                files_dir_exo = os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{exo_key}', st_sq_idx)

            boxes_exo = np.genfromtxt(os.path.join(files_dir_exo, 'boxes.txt'), delimiter=',')
            frame_idxs_exo = np.genfromtxt(os.path.join(files_dir_exo, 'frames.txt'), delimiter='\n', dtype=np.int64)
            visibilities_exo = np.genfromtxt(os.path.join(files_dir_exo, 'visibilities.txt'), delimiter='\n')

            assert frame_idxs_ego.shape[0] == frame_idxs_exo.shape[0]
            assert boxes_exo.shape == boxes_ego.shape

             

            if split == 'test':
                frame_idxs = list(range(frame_idxs_ego.min(), frame_idxs_ego.max()+1, 6))
            else:
                frame_idxs = list(frame_idxs_ego)

            images_ego = [os.path.join(self.frames_dir, f'{self.resolution}', 'takes', take_name, 'frame_aligned_videos', f'{ego_key}', f'{fi}.jpg') for fi in frame_idxs]
            images_exo = [os.path.join(self.frames_dir, f'{self.resolution}', 'takes', take_name, 'frame_aligned_videos', f'{exo_key}', f'{fi}.jpg') for fi in frame_idxs]

            boxes_ego_ = np.zeros((len(images_ego), 4)) + np.nan
            boxes_exo_ = np.zeros((len(images_exo), 4)) + np.nan
            for i, fi in enumerate(frame_idxs_ego):
                i_ = frame_idxs.index(fi)
                boxes_ego_[i_] = boxes_ego[i]
                boxes_exo_[i_] = boxes_exo[i]

            boxes_ego = boxes_ego_
            boxes_exo = boxes_exo_
            

            assert frame_idxs[0] == frame_idxs_ego[0] and frame_idxs[-1] == frame_idxs_ego[-1], f"Frame index mismatch: {frame_idxs[0]} != {frame_idxs_ego[0]} or {frame_idxs[-1]} != {frame_idxs_ego[-1]} for sequence {seq_name} ego"
            assert frame_idxs[0] == frame_idxs_exo[0] and frame_idxs[-1] == frame_idxs_exo[-1], f"Frame index mismatch: {frame_idxs[0]} != {frame_idxs_exo[0]} or {frame_idxs[-1]} != {frame_idxs_exo[-1]} for sequence {seq_name} exo"

            if self.mode == 'lt':
                #files_dir = os.path.join(self.frames_dir, resol_str, 'takes', take_name, 'frame_aligned_videos', f'{key}')
                masks_dir_ego = os.path.join(MASKS_DIR, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{ego_key}')
                masks_dir_exo = os.path.join(MASKS_DIR, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{exo_key}')

                img_files_ego = os.listdir(masks_dir_ego)
                img_idxs_ego = [int(img_file.split('.')[0]) for img_file in img_files_ego]
                img_idxs_ego = sorted(img_idxs_ego)

                img_files_exo = os.listdir(masks_dir_exo)
                img_idxs_exo = [int(img_file.split('.')[0]) for img_file in img_files_exo]
                img_idxs_exo = sorted(img_idxs_exo)
            else:
                #files_dir = os.path.join(self.frames_dir, resol_str, 'takes', take_name, 'frame_aligned_videos', f'{key}')
                masks_dir_ego = os.path.join(MASKS_DIR, 'single-obj', resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{ego_key}')
                masks_dir_exo = os.path.join(MASKS_DIR, 'single-obj', resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{exo_key}')
                img_idxs_ego = np.genfromtxt(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{ego_key}', st_sq_idx, 'frames.txt'), delimiter='\n', dtype=np.int64)
                img_idxs_exo = np.genfromtxt(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{exo_key}', st_sq_idx, 'frames.txt'), delimiter='\n', dtype=np.int64)

            #images = [f'{fi}.jpg' for fi in img_idxs]

            #images = [os.path.join(files_dir, f'{fi}.jpg') for fi in img_idxs]

            # load gt segmentations
            masks_path_ego = [os.path.join(masks_dir_ego, f'{fi}.png') for fi in img_idxs_ego]
            masks_path_exo = [os.path.join(masks_dir_exo, f'{fi}.png') for fi in img_idxs_exo]

            assert len(masks_path_ego) == len(masks_path_exo) == len(boxes_ego[~np.isnan(boxes_ego[:, 0])]) == len(boxes_exo[~np.isnan(boxes_exo[:, 0])])
            
            if split == 'test' and self.mode == 'lt':
                attribute_ego = {}
                attribute_exo = {}
                for attribute in attributes:
                    attribute_path_ego = os.path.join(ATTR_DIR, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{ego_key}', 'attributes', f'{attribute}.txt')
                    attributes_mask_ego = np.genfromtxt(attribute_path_ego, delimiter='\n')
                    attribute_ego[attribute] = attributes_mask_ego

                    attribute_path_exo = os.path.join(ATTR_DIR, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{exo_key}', 'attributes', f'{attribute}.txt')
                    attributes_mask_exo = np.genfromtxt(attribute_path_exo, delimiter='\n')
                    attribute_exo[attribute] = attributes_mask_exo
    

            ##### copy frames

            if copy_frames:
                if not os.path.exists(os.path.join(SAVE_DIR, 'frames', take_name, ego_key)):
                    os.makedirs(os.path.join(SAVE_DIR, 'frames', take_name, ego_key))
                if not os.path.exists(os.path.join(SAVE_DIR, 'frames', take_name, exo_key)):    
                    os.makedirs(os.path.join(SAVE_DIR, 'frames', take_name, exo_key))

                #print(seq_name)
                #print(images_ego)
                for img_file in images_ego:
                    shutil.copy(img_file, os.path.join(SAVE_DIR, 'frames', take_name, ego_key, os.path.basename(img_file)))

                assert all([os.path.exists(os.path.join(SAVE_DIR, 'frames', take_name, ego_key, os.path.basename(img_file))) for img_file in images_ego]), f"Not all frames were copied for {take_name} - {ego_key}. Check the download process."
                #assert len(images_ego) == len(os.listdir(os.path.join(SAVE_DIR, 'frames', take_name, ego_key))) == len(frame_idxs), (f"Not all frames are matching for {take_name} - {ego_key}.", (len(images_ego), len(os.listdir(os.path.join(SAVE_DIR, 'frames', take_name, ego_key))), len(frame_idxs))) 

                for img_file in images_exo:
                    shutil.copy(img_file, os.path.join(SAVE_DIR, 'frames', take_name, exo_key, os.path.basename(img_file)))

                assert all([os.path.exists(os.path.join(SAVE_DIR, 'frames', take_name, ego_key, os.path.basename(img_file))) for img_file in images_exo]), f"Not all frames were copied for {take_name} - {exo_key}. Check the download process."
                #assert len(images_exo) == len(os.listdir(os.path.join(SAVE_DIR, 'frames', take_name, exo_key))) == len(frame_idxs), (f"Not all frames are matching for {take_name} - {ego_key}.", (len(images_exo), len(os.listdir(os.path.join(SAVE_DIR, 'frames', take_name, exo_key))), len(frame_idxs))) 

            #### save annotations

            annotation_dict[seq_name] = {}
            annotation_dict[seq_name]['annotation_id'] = refactored_anno[seq_name]['annotation_id'] if self.mode == 'lt' else refactored_anno[st_seq_name]['annotation_id']
            annotation_dict[seq_name]['take'] = take_name
            annotation_dict[seq_name]['fpv_camera_name'] = ego_key
            annotation_dict[seq_name]['tpv_camera_name'] = exo_key

            annotation_dict[seq_name]['frame_annotations'] = {}
            for fi, frame_idx in enumerate(frame_idxs):
                anno_idx_ego = frame_idxs_ego.tolist().index(frame_idx) if frame_idx in frame_idxs_ego else -1
                anno_idx_exo = frame_idxs_exo.tolist().index(frame_idx) if frame_idx in frame_idxs_exo else -1

                assert anno_idx_ego == anno_idx_exo, f"Frame index mismatch: {anno_idx_ego} != {anno_idx_exo} for frame {frame_idx} in sequence {seq_name}"
                
                frame_idx = int(frame_idx)
                annotation_dict[seq_name]['frame_annotations'][frame_idx] = {}
                
                if not np.isnan(boxes_ego[fi]).any():
                    annotation_dict[seq_name]['frame_annotations'][frame_idx]['fpv'] = {}
                    annotation_dict[seq_name]['frame_annotations'][frame_idx]['fpv']['box'] = boxes_ego[fi].tolist()

                    encoded_mask = encode(np.asfortranarray(Image.open(masks_path_ego[anno_idx_ego])))
                    encoded_mask['counts'] = str(encoded_mask['counts'], "utf-8")
                    #print(encoded_mask)
                    annotation_dict[seq_name]['frame_annotations'][frame_idx]['fpv']['mask'] = encoded_mask
                    
                    # add attributes
                    if split == 'test' and self.mode == 'lt':
                        annotation_dict[seq_name]['frame_annotations'][frame_idx]['fpv']['attributes'] = []
                        for attribute in attributes:
                            if attribute_ego[attribute][anno_idx_ego] == 1:
                                annotation_dict[seq_name]['frame_annotations'][frame_idx]['fpv']['attributes'].append(attribute)
            
                
                if not np.isnan(boxes_exo[fi]).any():
                    annotation_dict[seq_name]['frame_annotations'][frame_idx]['tpv'] = {}
                    annotation_dict[seq_name]['frame_annotations'][frame_idx]['tpv']['box'] = boxes_exo[fi].tolist()

                    encoded_mask = encode(np.asfortranarray(Image.open(masks_path_exo[anno_idx_exo])))
                    encoded_mask['counts'] = str(encoded_mask['counts'], "utf-8")
                    #print(encoded_mask)
                    annotation_dict[seq_name]['frame_annotations'][frame_idx]['tpv']['mask'] = encoded_mask
                    
                    # add attributes
                    if split == 'test' and self.mode == 'lt':
                        annotation_dict[seq_name]['frame_annotations'][frame_idx]['tpv']['attributes'] = []
                        for attribute in attributes:
                            if attribute_exo[attribute][anno_idx_exo] == 1:
                                annotation_dict[seq_name]['frame_annotations'][frame_idx]['tpv']['attributes'].append(attribute)
            

        os.makedirs(os.path.join(SAVE_DIR, 'annotations'), exist_ok=True)
        # Save annotation_dict to a JSON file with pretty formatting
        save_path = os.path.join(SAVE_DIR, 'annotations', f"{split}_{self.mode}_annotations.json")
        with open(save_path, "w") as f:
            json.dump(annotation_dict, f, indent=4)
        print(f"Saved annotations to {save_path}")

