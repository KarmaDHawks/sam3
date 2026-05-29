from __future__ import absolute_import, division, print_function

import os
import numpy as np
import glob
import ast
import json
import time
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import cv2
import multiprocessing

from ..datasets import EgoExo4D
from ..utils.metrics import rect_iou, normalized_center_error, segm_iou, normalized_center_error_segm
from ..utils.viz import show_frame
from ..utils.ioutils import compress_file


class ExperimentEgoExo4D(object):
    r"""Experiment pipeline and evaluation toolkit for EgoExo4D dataset.
    
    Args:
        root_dir (string): Root directory of EgoExo4D dataset 
       
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, frames_dir, anno_dir, fps=30, resolution=720, mode='lt',
                 result_dir='results', report_dir='reports'):
        super(ExperimentEgoExo4D, self).__init__()
       
        
        #self.root_dir = root_dir
        self.fps = fps
        self.resolution = resolution
        self.mode = mode
        self.anno_dir = anno_dir
        self.dataset = EgoExo4D(frames_dir, anno_dir, fps=fps, resolution=resolution, mode=mode, download=False)

        self.masks_dir = '/media/TBDataNAS/Egocentric Vision/EgoExo4D/v2/annotations/vot_ego_exo/vos/val'


        self.result_dir = os.path.join(result_dir, 'VISTA')
        self.report_dir = os.path.join(report_dir, 'VISTA')
        self.nbins_iou = 21
        self.nbins_ce = 51
        self.nbins_nce = 51
        self.nbins_gsr = 51


    def run_sequence(self, idx, tracker, pov, visualize=False, save_video=False, overwrite_result=True):
        output =  self.dataset[idx]
        if output is None:
            print('In this sequence, the same frames are not annotated in both ego and exo, skipping', seq_name)
        else:
            img_files_ego, img_files_exo, anno_ego, anno_exo = output
            seq_name = self.dataset.seq_names[idx]
            print('--Sequence %d/%d: %s' % (
                idx + 1, len(self.dataset), seq_name))

            # skip if results exist
            if 'ego' in pov:
                record_dir_ego = os.path.join(
                    self.result_dir, tracker.name, self.mode, f'{self.fps}fps', str(self.resolution), seq_name, 'ego')
                if not os.path.exists(record_dir_ego):
                # tracking loop
                    boxes_ego, times_ego = tracker.track(
                        img_files_ego, anno_ego[0, :], anno=anno_ego, visualize=visualize)
                
                # record results
                    self._record(record_dir_ego, boxes_ego, times_ego)
                else:
                    print('  Found ego results, skipping', seq_name)

            if 'exo' in pov:    
                record_dir_exo = os.path.join(
                    self.result_dir, tracker.name, self.mode, f'{self.fps}fps', str(self.resolution), seq_name, 'exo')
                if not os.path.exists(record_dir_exo):
                # tracking loop
                    boxes_exo, confidences_exo, times_exo = tracker.track(
                        img_files_exo, anno_exo[0, :], anno=anno_exo, visualize=visualize)
                
                # record results
                    self._record(record_dir_exo, boxes_exo, confidences_exo, times_exo)
                else:
                    print('  Found exo results, skipping', seq_name)


    def run_sequential(self, tracker, pov, visualize=False, save_video=False, overwrite_result=True):
    
        print('Running tracker %s on EgoExo4D...' % tracker.name)
        self.dataset.return_meta = False

        # loop over the complete dataset
        #print(len(self.dataset))

        #for s, (img_files_ego, img_files_exo, anno_ego, anno_exo) in enumerate(self.dataset):
        for s in range(len(self.dataset)):
            self.run_sequence(s, tracker, pov, visualize=visualize)

    
    def run_parallel(self, tracker, pov, visualize=False, save_video=False, overwrite_result=True, threads=1):

        print('Running tracker %s on EgoExo4D...' % tracker.name)

        multiprocessing.set_start_method('spawn', force=True)

        param_list = [(seq_idx, tracker, pov) for seq_idx in range(len(self.dataset))]

        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(self.run_sequence, param_list)
    

    def report(self, tracker_names, pov):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        report_file = os.path.join(report_dir, f'performance-ope-{self.mode}-{pov}-{self.fps}fps.json')
        
        performance = {}
        for name in tracker_names:
            print('Evaluating', name, 'on', pov)

            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            norm_prec_curve = np.zeros((seq_num, self.nbins_nce))
            gen_succ_rob_curve = np.zeros((seq_num, self.nbins_gsr))
            speeds = np.zeros(seq_num)

            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, output in enumerate(self.dataset):
                _, _, anno_ego, anno_exo = output
                if pov == 'ego':
                    anno = anno_ego
                else:
                    anno = anno_exo
                seq_name = self.dataset.seq_names[s]
            
                record_file = os.path.join(
                    #self.result_dir, name, self.mode, f'{self.fps}fps', str(self.resolution), seq_name, pov, 'boxes.txt')
                    self.result_dir, name,'ope',seq_name +'.txt')
                
                if os.path.exists(record_file):
                    boxes = np.loadtxt(record_file, delimiter=',')
                else:
                    print('stiamo saltando',record_file)
                    continue 
                    
                
                if boxes.shape[0] != anno.shape[0]:
                    print(
                        f"[SKIP] mismatch boxes={boxes.shape[0]} anno={anno.shape[0]}"
                    )
                    continue

                boxes = boxes[~np.isnan(anno[:, 0])]
                anno = anno[~np.isnan(anno[:, 0])]

                boxes[0] = anno[0, :]
                assert len(boxes) == len(anno)

                #ious, center_errors = self._calc_metrics(boxes, anno[:, 1:])
                ious, norm_center_errors = self._calc_metrics(boxes, anno)
                succ_curve[s], norm_prec_curve[s] = self._calc_curves(ious, norm_center_errors)
                gen_succ_rob_curve[s] = self._calc_curves_robustness(ious)

                # calculate average tracking speed
               
                time_file = os.path.join(
                    self.result_dir, name, seq_name, pov, 'times.txt')
                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.mean(1. / times)
             
                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'success_curve': succ_curve[s].tolist(),
                    'normalized_precision_curve': norm_prec_curve[s].tolist(),
                    'generalized_success_robustness_curve': gen_succ_rob_curve[s].tolist(),
                    'success_score': np.mean(succ_curve[s]),
                    'normalized_precision_score': np.mean(norm_prec_curve[s]),
                    'generalized_success_robustness_score': np.mean(gen_succ_rob_curve[s]),
                    'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

            succ_curve = np.mean(succ_curve, axis=0)
            norm_prec_curve = np.mean(norm_prec_curve, axis=0)
            gen_succ_rob_curve = np.mean(gen_succ_rob_curve, axis=0)
            succ_score = np.mean(succ_curve)
            norm_prec_score = np.mean(norm_prec_curve)
            gen_succ_rob_score = np.mean(gen_succ_rob_curve)
            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'success_curve': succ_curve.tolist(),
                'normalized_precision_curve': norm_prec_curve.tolist(),
                'generalized_success_robustness_curve': gen_succ_rob_curve.tolist(),
                'success_score': succ_score,
                'normalized_precision_score': norm_prec_score,
                'generalized_success_robustness_score': gen_succ_rob_score})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        #if not realtime:
        #    self.plot_curves(tracker_names)

        return performance


    def report_vot(self, tracker_names, pov):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        report_file = os.path.join(report_dir, f'performance-vot-ope-{self.mode}-{pov}-{self.fps}fps.json')
        
        performance = {}
        for name in tracker_names:
            print('Evaluating', name, 'on', pov)

            seq_num = len(self.dataset)
            all_iou = np.zeros(seq_num)
            all_nps = np.zeros(seq_num)
            all_gsr = np.zeros(seq_num)
            weights = np.zeros(seq_num)

            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, output in enumerate(self.dataset):
                _, _, anno_ego, anno_exo = output
                if pov == 'ego':
                    anno = anno_ego
                else:
                    anno = anno_exo
                seq_name = self.dataset.seq_names[s]
            
                record_file = os.path.join(
                    self.result_dir, name, self.mode, f'{self.fps}fps', str(self.resolution), seq_name, pov, 'boxes.txt')

                boxes = np.loadtxt(record_file, delimiter=',')
                boxes = boxes[~np.isnan(anno[:, 0])]
                anno = anno[~np.isnan(anno[:, 0])]

                boxes[0] = anno[0, :]
                assert len(boxes) == len(anno)

                #ious, center_errors = self._calc_metrics(boxes, anno[:, 1:])
                ious, norm_center_errors = self._calc_metrics(boxes, anno)
                succ_curve, norm_prec_curve = self._calc_curves(ious, norm_center_errors)
                gen_succ_rob_curve = self._calc_curves_robustness(ious)


                all_iou[s] = np.mean(succ_curve)
                all_nps[s] = np.mean(norm_prec_curve)
                all_gsr[s] = np.mean(gen_succ_rob_curve)
                weights[s] = len(anno)

                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'success_score': np.mean(succ_curve),
                    'normalized_precision_score': np.mean(norm_prec_curve),
                    'generalized_success_robustness_score': np.mean(gen_succ_rob_curve)
                    }})

            avg_iou = (all_iou * weights).sum() / np.sum(weights)
            avg_nps = (all_nps * weights).sum() / np.sum(weights)
            avg_gsr = (all_gsr * weights).sum() / np.sum(weights)

            print(avg_iou, avg_nps, avg_gsr)
            
            # store overall performance
            performance[name]['overall'].update({
                'avg_iou': avg_iou,
                'avg_normalized_precision_score': avg_nps,
                'avg_generalized_success_robustness_score': avg_gsr})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        #if not realtime:
        #    self.plot_curves(tracker_names)

        return performance


    def report_per_attribute(self, tracker_names, pov, attribute):
        assert isinstance(tracker_names, (list, tuple))

        attributes_dir = f'/media/TBDataNAS/Egocentric Vision/EgoExo4D/v2/annotations/vot_ego_exo/sot/val/lt/'
        attributes_dir = attributes_dir + f"{self.resolution}" if self.resolution > 0 else attributes_dir

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0], 'attributes')
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        report_file = os.path.join(report_dir, f'performance-ope-{self.mode}-{pov}-{self.fps}fps-{attribute}.json')
        
        performance = {}
        for name in tracker_names:
            print('Evaluating', name, 'on', pov)

            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            norm_prec_curve = np.zeros((seq_num, self.nbins_nce))
            gen_succ_rob_curve = np.zeros((seq_num, self.nbins_gsr))
            speeds = np.zeros(seq_num)

            seqs_processed = np.zeros(seq_num)

            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, output in enumerate(self.dataset):
                _, _, anno_ego, anno_exo = output
                if pov == 'ego':
                    anno = anno_ego
                else:
                    anno = anno_exo
                seq_name = self.dataset.seq_names[s]
            
                record_file = os.path.join(
                    self.result_dir, name, self.mode, f'{self.fps}fps', str(self.resolution), seq_name, pov, 'boxes.txt')

                take_name = seq_name.split('*')[0]
                if self.mode == 'lt':
                    keys = sorted(os.listdir(os.path.join(attributes_dir, 'takes', take_name, seq_name, 'frame_aligned_videos')))
                else:
                    keys = sorted(os.listdir(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos')))
                #print(keys, seq_name)
                ego_key = keys[0]
                exo_key = keys[1]
                assert 'aria' in ego_key, keys
                if pov == 'ego':
                    key = ego_key
                else:
                    key = exo_key
                attribute_path = os.path.join(attributes_dir, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{key}', 'attributes', f'{attribute}.txt')
                attributes_mask = np.genfromtxt(attribute_path, delimiter='\n')

                
                boxes = np.loadtxt(record_file, delimiter=',')
                boxes = boxes[~np.isnan(anno[:, 0])]
                anno = anno[~np.isnan(anno[:, 0])]

                assert len(attributes_mask) == anno.shape[0], (len(attributes_mask), anno.shape[0])

                boxes = boxes[attributes_mask > 0]
                anno = anno[attributes_mask > 0]

                if attributes_mask.sum() > 0:
                    seqs_processed[s] = 1

                    boxes[0] = anno[0, :]
                    assert len(boxes) == len(anno)

                    #ious, center_errors = self._calc_metrics(boxes, anno[:, 1:])
                    ious, norm_center_errors = self._calc_metrics(boxes, anno)
                    succ_curve[s], norm_prec_curve[s] = self._calc_curves(ious, norm_center_errors)
                    gen_succ_rob_curve[s] = self._calc_curves_robustness(ious)

                    # calculate average tracking speed
                    time_file = os.path.join(
                        self.result_dir, name, seq_name, pov, 'times.txt')
                    if os.path.isfile(time_file):
                        times = np.loadtxt(time_file)
                        times = times[times > 0]
                        if len(times) > 0:
                            speeds[s] = np.mean(1. / times)

                    # store sequence-wise performance
                    performance[name]['seq_wise'].update({seq_name: {
                        'success_curve': succ_curve[s].tolist(),
                        'normalized_precision_curve': norm_prec_curve[s].tolist(),
                        'generalized_success_robustness_curve': gen_succ_rob_curve[s].tolist(),
                        'success_score': np.mean(succ_curve[s]),
                        'normalized_precision_score': np.mean(norm_prec_curve[s]),
                        'generalized_success_robustness_score': np.mean(gen_succ_rob_curve[s]),
                        'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

            succ_curve = np.mean(succ_curve[seqs_processed > 0], axis=0)
            norm_prec_curve = np.mean(norm_prec_curve[seqs_processed > 0], axis=0)
            gen_succ_rob_curve = np.mean(gen_succ_rob_curve[seqs_processed > 0], axis=0)
            succ_score = np.mean(succ_curve)
            norm_prec_score = np.mean(norm_prec_curve)
            gen_succ_rob_score = np.mean(gen_succ_rob_curve)
            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'success_curve': succ_curve.tolist(),
                'normalized_precision_curve': norm_prec_curve.tolist(),
                'generalized_success_robustness_curve': gen_succ_rob_curve.tolist(),
                'success_score': succ_score,
                'normalized_precision_score': norm_prec_score,
                'generalized_success_robustness_score': gen_succ_rob_score,
                'speed_fps': avg_speed})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        #if not realtime:
        #    self.plot_curves(tracker_names)

        return performance


    def report_per_attribute_vot(self, tracker_names, pov, attribute, redet=None):
        assert isinstance(tracker_names, (list, tuple))

        attributes_dir = f'/media/TBDataNAS/Egocentric Vision/EgoExo4D/v2/annotations/vot_ego_exo/sot/val/lt/'
        attributes_dir = attributes_dir + f"{self.resolution}" if self.resolution > 0 else attributes_dir

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0], 'attributes')
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        if redet is None:
            report_file = os.path.join(report_dir, f'performance-vot-ope-{self.mode}-{pov}-{self.fps}fps-{attribute}.json')
        else:
            report_file = os.path.join(report_dir, f'performance-vot-ope-{self.mode}-{pov}-{self.fps}fps-{attribute}-redet_{redet}.json')
        
        performance = {}
        for name in tracker_names:
            print('Evaluating', name, 'on', pov)

            seq_num = len(self.dataset)
            all_iou = np.zeros(seq_num)
            all_nps = np.zeros(seq_num)
            all_gsr = np.zeros(seq_num)
            weights = np.zeros(seq_num)

            seqs_processed = np.zeros(seq_num)

            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, output in enumerate(self.dataset):
                _, _, anno_ego, anno_exo = output
                if pov == 'ego':
                    anno = anno_ego
                else:
                    anno = anno_exo
                seq_name = self.dataset.seq_names[s]
            
                record_file = os.path.join(
                    self.result_dir, name, self.mode, f'{self.fps}fps', str(self.resolution), seq_name, pov, 'boxes.txt')

                take_name = seq_name.split('*')[0]
                if self.mode == 'lt':
                    keys = sorted(os.listdir(os.path.join(attributes_dir, 'takes', take_name, seq_name, 'frame_aligned_videos')))
                else:
                    keys = sorted(os.listdir(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos')))
                #print(keys, seq_name)
                ego_key = keys[0]
                exo_key = keys[1]
                assert 'aria' in ego_key, keys
                if pov == 'ego':
                    key = ego_key
                else:
                    key = exo_key
                attribute_path = os.path.join(attributes_dir, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{key}', 'attributes', f'{attribute}.txt')
                attributes_mask = np.genfromtxt(attribute_path, delimiter='\n')
                
                if redet is not None:
                    redet_path = os.path.join(attributes_dir, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{key}', 'attributes', f'redet_periods_{redet}.txt')
                    redet_mask = np.genfromtxt(redet_path, delimiter='\n')

                    attributes_mask = attributes_mask * redet_mask

                
                #print(attributes_mask)
                boxes = np.loadtxt(record_file, delimiter=',')
                boxes = boxes[~np.isnan(anno[:, 0])]
                anno = anno[~np.isnan(anno[:, 0])]

                assert len(attributes_mask) == anno.shape[0], (len(attributes_mask), anno.shape[0])

                boxes = boxes[attributes_mask > 0]
                anno = anno[attributes_mask > 0]

                if attributes_mask.sum() > 0:
                    seqs_processed[s] = 1

                    boxes[0] = anno[0, :]
                    assert len(boxes) == len(anno)

                    #ious, center_errors = self._calc_metrics(boxes, anno[:, 1:])
                    ious, norm_center_errors = self._calc_metrics(boxes, anno)
                    succ_curve, norm_prec_curve = self._calc_curves(ious, norm_center_errors)
                    gen_succ_rob_curve = self._calc_curves_robustness(ious)

                    all_iou[s] = np.mean(succ_curve)
                    all_nps[s] = np.mean(norm_prec_curve)
                    all_gsr[s] = np.mean(gen_succ_rob_curve)
                    weights[s] = len(anno)

                    # store sequence-wise performance
                    performance[name]['seq_wise'].update({seq_name: {
                        'success_score': np.mean(succ_curve),
                        'normalized_precision_score': np.mean(norm_prec_curve),
                        'generalized_success_robustness_score': np.mean(gen_succ_rob_curve)}})


            avg_iou = (all_iou * weights).sum() / np.sum(weights)
            avg_nps = (all_nps * weights).sum() / np.sum(weights)
            avg_gsr = (all_gsr * weights).sum() / np.sum(weights)

            print(avg_iou, avg_nps, avg_gsr)
            
            # store overall performance
            performance[name]['overall'].update({
                'avg_iou': avg_iou,
                'avg_normalized_precision_score': avg_nps,
                'avg_generalized_success_robustness_score': avg_gsr})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        #if not realtime:
        #    self.plot_curves(tracker_names)

        return performance


    def report_segm(self, name, results_dir, pov):
        #assert isinstance(name, (list, tuple))

        
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, name)
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        report_file = os.path.join(report_dir, f'performance-ope-{self.mode}-{pov}-{self.fps}fps.json')
        
        performance = {}

        print('Evaluating', name, 'on', pov)

        seq_num = len(self.dataset)
        succ_curve = np.zeros((seq_num, self.nbins_iou))
        norm_prec_curve = np.zeros((seq_num, self.nbins_nce))
        gen_succ_rob_curve = np.zeros((seq_num, self.nbins_gsr))
        speeds = np.zeros(seq_num)

        performance.update({name: {
            'overall': {},
            'seq_wise': {}}})

        for s, output in enumerate(self.dataset):
            _, _, anno_ego, anno_exo = output
            #if pov == 'ego':
            #    anno = anno_ego
            #else:
            #    anno = anno_exo
            seq_name = self.dataset.seq_names[s]

            print(f'Processing {s+1}/{len(self.dataset)}')

            if self.mode == 'st':
                seq_name_parts = seq_name.split('$')
                st_sq_idx = seq_name_parts[-1]
                st_seq_name = seq_name_parts[0]

            take_name = seq_name.split('*')[0]
        
            #record_file = os.path.join(
            #    self.result_dir, name, self.mode, f'{self.fps}fps', str(self.resolution), seq_name, pov, 'boxes.txt')

            #boxes = np.loadtxt(record_file, delimiter=',')
            #boxes = boxes[~np.isnan(anno[:, 0])]
            #anno = anno[~np.isnan(anno[:, 0])]


            resol_str = f"{self.resolution}" if self.resolution > 0 else ""
            
            #keys = sorted(os.listdir(os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos')))

            if self.mode == 'lt':
                keys = sorted(os.listdir(os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos')))
            else:
                keys = sorted(os.listdir(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos')))
            #print(keys, seq_name)
            #print(keys, seq_name)
            if pov == 'ego':
                key = keys[0]
            else:
                key = keys[1]

            if self.mode == 'lt':
                #files_dir = os.path.join(self.frames_dir, resol_str, 'takes', take_name, 'frame_aligned_videos', f'{key}')
                masks_dir = os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{key}')

                img_files = os.listdir(masks_dir)
                img_idxs = [int(img_file.split('.')[0]) for img_file in img_files]
                img_idxs = sorted(img_idxs)
            else:
                #files_dir = os.path.join(self.frames_dir, resol_str, 'takes', take_name, 'frame_aligned_videos', f'{key}')
                masks_dir = os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{key}')
                img_idxs = np.genfromtxt(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{key}', st_sq_idx, 'frames.txt'), delimiter='\n', dtype=np.int64)

            #images = [f'{fi}.jpg' for fi in img_idxs]

            #images = [os.path.join(files_dir, f'{fi}.jpg') for fi in img_idxs]

            # load gt segmentations
            masks_path = [os.path.join(masks_dir, f'{fi}.png') for fi in img_idxs]

            assert len(masks_path) == len(anno_ego[~np.isnan(anno_ego[:, 0])]) == len(anno_exo[~np.isnan(anno_exo[:, 0])])
            
        
    
            anno_segm = []
            for mp in masks_path:
                mask = np.array(Image.open(mp))
                anno_segm.append(mask)
            anno_segm = np.array(anno_segm)

            seq_pred_dir = os.path.join(results_dir, pov, seq_name)
            pred_paths = [os.path.join(seq_pred_dir, f'{fi}.png') for fi in img_idxs]

            # load predicted segmentations
            segm = []
            for pp in pred_paths:
                #print(np.unique(np.array(Image.open(pp))))
                mask = np.array(Image.open(pp))
                segm.append(mask)
            segm = np.array(segm)

            if name == 'AOT' and segm.shape != anno_segm.shape:
                segm = np.zeros_like(anno_segm)

            segm[0] = anno_segm[0]
            assert len(segm) == len(anno_segm)

            

            #ious, center_errors = self._calc_metrics(boxes, anno[:, 1:])
            ious, norm_center_errors = self._calc_metrics_segm(segm, anno_segm)
            #print(ious)
            succ_curve[s], norm_prec_curve[s] = self._calc_curves(ious, norm_center_errors)
            gen_succ_rob_curve[s] = self._calc_curves_robustness(ious)

            # calculate average tracking speed
            time_file = os.path.join(
                self.result_dir, name, seq_name, pov, 'times.txt')
            if os.path.isfile(time_file):
                times = np.loadtxt(time_file)
                times = times[times > 0]
                if len(times) > 0:
                    speeds[s] = np.mean(1. / times)

            # store sequence-wise performance
            performance[name]['seq_wise'].update({seq_name: {
                'success_curve': succ_curve[s].tolist(),
                'normalized_precision_curve': norm_prec_curve[s].tolist(),
                'generalized_success_robustness_curve': gen_succ_rob_curve[s].tolist(),
                'success_score': np.mean(succ_curve[s]),
                'normalized_precision_score': np.mean(norm_prec_curve[s]),
                'generalized_success_robustness_score': np.mean(gen_succ_rob_curve[s]),
                'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

        succ_curve = np.mean(succ_curve, axis=0)
        norm_prec_curve = np.mean(norm_prec_curve, axis=0)
        gen_succ_rob_curve = np.mean(gen_succ_rob_curve, axis=0)
        succ_score = np.mean(succ_curve)
        norm_prec_score = np.mean(norm_prec_curve)
        gen_succ_rob_score = np.mean(gen_succ_rob_curve)
        if np.count_nonzero(speeds) > 0:
            avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
        else:
            avg_speed = -1

        # store overall performance
        performance[name]['overall'].update({
            'success_curve': succ_curve.tolist(),
            'normalized_precision_curve': norm_prec_curve.tolist(),
            'generalized_success_robustness_curve': gen_succ_rob_curve.tolist(),
            'success_score': succ_score,
            'normalized_precision_score': norm_prec_score,
            'generalized_success_robustness_score': gen_succ_rob_score,
            'speed_fps': avg_speed})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        #if not realtime:
        #    self.plot_curves(tracker_names)

        return performance


    def report_vot_segm(self, name, results_dir, pov):
        #assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, name)
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        report_file = os.path.join(report_dir, f'performance-vot-ope-{self.mode}-{pov}-{self.fps}fps.json')
        
        performance = {}
        
        print('Evaluating', name, 'on', pov)

        seq_num = len(self.dataset)
        all_iou = np.zeros(seq_num)
        all_nps = np.zeros(seq_num)
        all_gsr = np.zeros(seq_num)
        weights = np.zeros(seq_num)

        performance.update({name: {
            'overall': {},
            'seq_wise': {}}})

        for s, output in enumerate(self.dataset):
            _, _, anno_ego, anno_exo = output
            
            seq_name = self.dataset.seq_names[s]
        
            if self.mode == 'st':
                seq_name_parts = seq_name.split('$')
                st_sq_idx = seq_name_parts[-1]
                st_seq_name = seq_name_parts[0]

            take_name = seq_name.split('*')[0]
        
            #record_file = os.path.join(
            #    self.result_dir, name, self.mode, f'{self.fps}fps', str(self.resolution), seq_name, pov, 'boxes.txt')

            #boxes = np.loadtxt(record_file, delimiter=',')
            #boxes = boxes[~np.isnan(anno[:, 0])]
            #anno = anno[~np.isnan(anno[:, 0])]


            resol_str = f"{self.resolution}" if self.resolution > 0 else ""
            
            #keys = sorted(os.listdir(os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos')))

            if self.mode == 'lt':
                keys = sorted(os.listdir(os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos')))
            else:
                keys = sorted(os.listdir(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos')))
            #print(keys, seq_name)
            #print(keys, seq_name)
            if pov == 'ego':
                key = keys[0]
            else:
                key = keys[1]

            if self.mode == 'lt':
                #files_dir = os.path.join(self.frames_dir, resol_str, 'takes', take_name, 'frame_aligned_videos', f'{key}')
                masks_dir = os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{key}')

                img_files = os.listdir(masks_dir)
                img_idxs = [int(img_file.split('.')[0]) for img_file in img_files]
                img_idxs = sorted(img_idxs)
            else:
                #files_dir = os.path.join(self.frames_dir, resol_str, 'takes', take_name, 'frame_aligned_videos', f'{key}')
                masks_dir = os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{key}')
                img_idxs = np.genfromtxt(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{key}', st_sq_idx, 'frames.txt'), delimiter='\n', dtype=np.int64)

            #images = [f'{fi}.jpg' for fi in img_idxs]

            #images = [os.path.join(files_dir, f'{fi}.jpg') for fi in img_idxs]

            # load gt segmentations
            masks_path = [os.path.join(masks_dir, f'{fi}.png') for fi in img_idxs]

            assert len(masks_path) == len(anno_ego[~np.isnan(anno_ego[:, 0])]) == len(anno_exo[~np.isnan(anno_exo[:, 0])])
            
        
    
            anno_segm = []
            for mp in masks_path:
                mask = np.array(Image.open(mp))
                anno_segm.append(mask)
            anno_segm = np.array(anno_segm)

            seq_pred_dir = os.path.join(results_dir, pov, seq_name)
            pred_paths = [os.path.join(seq_pred_dir, f'{fi}.png') for fi in img_idxs]

            # load predicted segmentations
            segm = []
            for pp in pred_paths:
                #print(np.unique(np.array(Image.open(pp))))
                mask = np.array(Image.open(pp))
                segm.append(mask)
            segm = np.array(segm)

            if name == 'AOT' and segm.shape != anno_segm.shape:
                segm = np.zeros_like(anno_segm)

            segm[0] = anno_segm[0]
            assert len(segm) == len(anno_segm)

            #ious, center_errors = self._calc_metrics(boxes, anno[:, 1:])
            ious, norm_center_errors = self._calc_metrics_segm(segm, anno_segm)
            succ_curve, norm_prec_curve = self._calc_curves(ious, norm_center_errors)
            gen_succ_rob_curve = self._calc_curves_robustness(ious)


            all_iou[s] = np.mean(succ_curve)
            all_nps[s] = np.mean(norm_prec_curve)
            all_gsr[s] = np.mean(gen_succ_rob_curve)
            weights[s] = len(anno_segm)

            # store sequence-wise performance
            performance[name]['seq_wise'].update({seq_name: {
                'success_score': np.mean(succ_curve),
                'normalized_precision_score': np.mean(norm_prec_curve),
                'generalized_success_robustness_score': np.mean(gen_succ_rob_curve)
                }})

        avg_iou = (all_iou * weights).sum() / np.sum(weights)
        avg_nps = (all_nps * weights).sum() / np.sum(weights)
        avg_gsr = (all_gsr * weights).sum() / np.sum(weights)

        print(avg_iou, avg_nps, avg_gsr)
        
        # store overall performance
        performance[name]['overall'].update({
            'avg_iou': avg_iou,
            'avg_normalized_precision_score': avg_nps,
            'avg_generalized_success_robustness_score': avg_gsr})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        #if not realtime:
        #    self.plot_curves(tracker_names)

        return performance



    def report_segm_per_attribute(self, name, results_dir, pov, attribute):
        #assert isinstance(tracker_names, (list, tuple))

        attributes_dir = f'/media/TBDataNAS/Egocentric Vision/EgoExo4D/v2/annotations/vot_ego_exo/sot/val/lt/'
        attributes_dir = attributes_dir + f"{self.resolution}" if self.resolution > 0 else attributes_dir

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, name, 'attributes')
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        report_file = os.path.join(report_dir, f'performance-ope-{self.mode}-{pov}-{self.fps}fps-{attribute}.json')
        
        performance = {}

        print('Evaluating', name, 'on', pov)

        seq_num = len(self.dataset)
        succ_curve = np.zeros((seq_num, self.nbins_iou))
        norm_prec_curve = np.zeros((seq_num, self.nbins_nce))
        gen_succ_rob_curve = np.zeros((seq_num, self.nbins_gsr))
        speeds = np.zeros(seq_num)

        seqs_processed = np.zeros(seq_num)

        performance.update({name: {
            'overall': {},
            'seq_wise': {}}})

        for s, output in enumerate(self.dataset):
            _, _, anno_ego, anno_exo = output
            #if pov == 'ego':
            #    anno = anno_ego
            #else:
            #    anno = anno_exo
            seq_name = self.dataset.seq_names[s]

            print(f'Processing {s+1}/{len(self.dataset)}')

            if self.mode == 'st':
                seq_name_parts = seq_name.split('$')
                st_sq_idx = seq_name_parts[-1]
                st_seq_name = seq_name_parts[0]

            take_name = seq_name.split('*')[0]
        
            #record_file = os.path.join(
            #    self.result_dir, name, self.mode, f'{self.fps}fps', str(self.resolution), seq_name, pov, 'boxes.txt')

            #boxes = np.loadtxt(record_file, delimiter=',')
            #boxes = boxes[~np.isnan(anno[:, 0])]
            #anno = anno[~np.isnan(anno[:, 0])]


            resol_str = f"{self.resolution}" if self.resolution > 0 else ""
            
            #keys = sorted(os.listdir(os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos')))

            if self.mode == 'lt':
                keys = sorted(os.listdir(os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos')))
            else:
                keys = sorted(os.listdir(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos')))
            #print(keys, seq_name)
            #print(keys, seq_name)
            if pov == 'ego':
                key = keys[0]
            else:
                key = keys[1]

            if self.mode == 'lt':
                #files_dir = os.path.join(self.frames_dir, resol_str, 'takes', take_name, 'frame_aligned_videos', f'{key}')
                masks_dir = os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{key}')

                img_files = os.listdir(masks_dir)
                img_idxs = [int(img_file.split('.')[0]) for img_file in img_files]
                img_idxs = sorted(img_idxs)
            else:
                #files_dir = os.path.join(self.frames_dir, resol_str, 'takes', take_name, 'frame_aligned_videos', f'{key}')
                masks_dir = os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{key}')
                img_idxs = np.genfromtxt(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{key}', st_sq_idx, 'frames.txt'), delimiter='\n', dtype=np.int64)

            #images = [f'{fi}.jpg' for fi in img_idxs]

            #images = [os.path.join(files_dir, f'{fi}.jpg') for fi in img_idxs]

            attribute_path = os.path.join(attributes_dir, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{key}', 'attributes', f'{attribute}.txt')
            attributes_mask = np.genfromtxt(attribute_path, delimiter='\n')

            img_idxs = np.array(img_idxs)
            img_idxs = img_idxs[attributes_mask > 0]

            

            # load gt segmentations
            masks_path = [os.path.join(masks_dir, f'{fi}.png') for fi in img_idxs]

            assert len(masks_path) == len(anno_ego[~np.isnan(anno_ego[:, 0])][attributes_mask > 0]) == len(anno_exo[~np.isnan(anno_exo[:, 0])][attributes_mask > 0])

            #masks_path = np.array(masks_path)

            if attributes_mask.sum() > 0:
                seqs_processed[s] = 1
            
                anno_segm = []
                for mp in masks_path:
                    mask = np.array(Image.open(mp))
                    anno_segm.append(mask)
                anno_segm = np.array(anno_segm)

                seq_pred_dir = os.path.join(results_dir, pov, seq_name)
                pred_paths = [os.path.join(seq_pred_dir, f'{fi}.png') for fi in img_idxs]

                # load predicted segmentations
                segm = []
                for pp in pred_paths:
                    #print(np.unique(np.array(Image.open(pp))))
                    mask = np.array(Image.open(pp))
                    segm.append(mask)
                segm = np.array(segm)

                if name == 'AOT' and segm.shape != anno_segm.shape:
                    segm = np.zeros_like(anno_segm)

                segm[0] = anno_segm[0]
                assert len(segm) == len(anno_segm)

                

                #ious, center_errors = self._calc_metrics(boxes, anno[:, 1:])
                ious, norm_center_errors = self._calc_metrics_segm(segm, anno_segm)
                #print(ious)
                succ_curve[s], norm_prec_curve[s] = self._calc_curves(ious, norm_center_errors)
                gen_succ_rob_curve[s] = self._calc_curves_robustness(ious)

                # calculate average tracking speed
                time_file = os.path.join(
                    self.result_dir, name, seq_name, pov, 'times.txt')
                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.mean(1. / times)

                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'success_curve': succ_curve[s].tolist(),
                    'normalized_precision_curve': norm_prec_curve[s].tolist(),
                    'generalized_success_robustness_curve': gen_succ_rob_curve[s].tolist(),
                    'success_score': np.mean(succ_curve[s]),
                    'normalized_precision_score': np.mean(norm_prec_curve[s]),
                    'generalized_success_robustness_score': np.mean(gen_succ_rob_curve[s]),
                    'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

        succ_curve = np.mean(succ_curve[seqs_processed > 0], axis=0)
        norm_prec_curve = np.mean(norm_prec_curve[seqs_processed > 0], axis=0)
        gen_succ_rob_curve = np.mean(gen_succ_rob_curve[seqs_processed > 0], axis=0)
        succ_score = np.mean(succ_curve)
        norm_prec_score = np.mean(norm_prec_curve)
        gen_succ_rob_score = np.mean(gen_succ_rob_curve)
        if np.count_nonzero(speeds) > 0:
            avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
        else:
            avg_speed = -1

        # store overall performance
        performance[name]['overall'].update({
            'success_curve': succ_curve.tolist(),
            'normalized_precision_curve': norm_prec_curve.tolist(),
            'generalized_success_robustness_curve': gen_succ_rob_curve.tolist(),
            'success_score': succ_score,
            'normalized_precision_score': norm_prec_score,
            'generalized_success_robustness_score': gen_succ_rob_score,
            'speed_fps': avg_speed})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        #if not realtime:
        #    self.plot_curves(tracker_names)

        return performance


    def report_segm_per_attribute_vot(self, name, results_dir, pov, attribute, redet=None):
        #assert isinstance(tracker_names, (list, tuple))

        attributes_dir = f'/media/TBDataNAS/Egocentric Vision/EgoExo4D/v2/annotations/vot_ego_exo/sot/val/lt/'
        attributes_dir = attributes_dir + f"{self.resolution}" if self.resolution > 0 else attributes_dir

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, name, 'attributes')
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        if redet is None:
            report_file = os.path.join(report_dir, f'performance-vot-ope-{self.mode}-{pov}-{self.fps}fps-{attribute}.json')
        else:
            report_file = os.path.join(report_dir, f'performance-vot-ope-{self.mode}-{pov}-{self.fps}fps-{attribute}-redet_{redet}.json')
        
        performance = {}

        print('Evaluating', name, 'on', pov)

        seq_num = len(self.dataset)
        all_iou = np.zeros(seq_num)
        all_nps = np.zeros(seq_num)
        all_gsr = np.zeros(seq_num)
        weights = np.zeros(seq_num)

        seqs_processed = np.zeros(seq_num)

        performance.update({name: {
            'overall': {},
            'seq_wise': {}}})

        for s, output in enumerate(self.dataset):
            _, _, anno_ego, anno_exo = output
            #if pov == 'ego':
            #    anno = anno_ego
            #else:
            #    anno = anno_exo
            seq_name = self.dataset.seq_names[s]

            print(f'Processing {s+1}/{len(self.dataset)}')

            if self.mode == 'st':
                seq_name_parts = seq_name.split('$')
                st_sq_idx = seq_name_parts[-1]
                st_seq_name = seq_name_parts[0]

            take_name = seq_name.split('*')[0]
        
            #record_file = os.path.join(
            #    self.result_dir, name, self.mode, f'{self.fps}fps', str(self.resolution), seq_name, pov, 'boxes.txt')

            #boxes = np.loadtxt(record_file, delimiter=',')
            #boxes = boxes[~np.isnan(anno[:, 0])]
            #anno = anno[~np.isnan(anno[:, 0])]


            resol_str = f"{self.resolution}" if self.resolution > 0 else ""
            
            #keys = sorted(os.listdir(os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos')))

            if self.mode == 'lt':
                keys = sorted(os.listdir(os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos')))
            else:
                keys = sorted(os.listdir(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos')))
            #print(keys, seq_name)
            #print(keys, seq_name)
            if pov == 'ego':
                key = keys[0]
            else:
                key = keys[1]

            if self.mode == 'lt':
                #files_dir = os.path.join(self.frames_dir, resol_str, 'takes', take_name, 'frame_aligned_videos', f'{key}')
                masks_dir = os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{key}')

                img_files = os.listdir(masks_dir)
                img_idxs = [int(img_file.split('.')[0]) for img_file in img_files]
                img_idxs = sorted(img_idxs)
            else:
                #files_dir = os.path.join(self.frames_dir, resol_str, 'takes', take_name, 'frame_aligned_videos', f'{key}')
                masks_dir = os.path.join(self.masks_dir, 'single-obj', resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{key}')
                img_idxs = np.genfromtxt(os.path.join(self.anno_dir, self.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{key}', st_sq_idx, 'frames.txt'), delimiter='\n', dtype=np.int64)

            #images = [f'{fi}.jpg' for fi in img_idxs]

            #images = [os.path.join(files_dir, f'{fi}.jpg') for fi in img_idxs]

            attribute_path = os.path.join(attributes_dir, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{key}', 'attributes', f'{attribute}.txt')
            attributes_mask = np.genfromtxt(attribute_path, delimiter='\n')

            if redet is not None:
                redet_path = os.path.join(attributes_dir, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{key}', 'attributes', f'redet_periods_{redet}.txt')
                redet_mask = np.genfromtxt(redet_path, delimiter='\n')

                attributes_mask = attributes_mask * redet_mask

            img_idxs = np.array(img_idxs)
            img_idxs = img_idxs[attributes_mask > 0]

            

            # load gt segmentations
            masks_path = [os.path.join(masks_dir, f'{fi}.png') for fi in img_idxs]

            assert len(masks_path) == len(anno_ego[~np.isnan(anno_ego[:, 0])][attributes_mask > 0]) == len(anno_exo[~np.isnan(anno_exo[:, 0])][attributes_mask > 0])

            #masks_path = np.array(masks_path)

            if attributes_mask.sum() > 0:
                seqs_processed[s] = 1
            
                anno_segm = []
                for mp in masks_path:
                    mask = np.array(Image.open(mp))
                    anno_segm.append(mask)
                anno_segm = np.array(anno_segm)

                seq_pred_dir = os.path.join(results_dir, pov, seq_name)
                pred_paths = [os.path.join(seq_pred_dir, f'{fi}.png') for fi in img_idxs]

                # load predicted segmentations
                segm = []
                for pp in pred_paths:
                    #print(np.unique(np.array(Image.open(pp))))
                    mask = np.array(Image.open(pp))
                    segm.append(mask)
                segm = np.array(segm)

                if name == 'AOT' and segm.shape != anno_segm.shape:
                    segm = np.zeros_like(anno_segm)

                segm[0] = anno_segm[0]
                assert len(segm) == len(anno_segm)

                

                #ious, center_errors = self._calc_metrics(boxes, anno[:, 1:])
                ious, norm_center_errors = self._calc_metrics_segm(segm, anno_segm)
                #print(ious)
                succ_curve, norm_prec_curve = self._calc_curves(ious, norm_center_errors)
                gen_succ_rob_curve = self._calc_curves_robustness(ious)

                all_iou[s] = np.mean(succ_curve)
                all_nps[s] = np.mean(norm_prec_curve)
                all_gsr[s] = np.mean(gen_succ_rob_curve)
                weights[s] = len(anno_segm)

                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'success_score': np.mean(succ_curve),
                    'normalized_precision_score': np.mean(norm_prec_curve),
                    'generalized_success_robustness_score': np.mean(gen_succ_rob_curve)}})

        avg_iou = (all_iou * weights).sum() / np.sum(weights)
        avg_nps = (all_nps * weights).sum() / np.sum(weights)
        avg_gsr = (all_gsr * weights).sum() / np.sum(weights)

        #print(avg_iou, avg_nps, avg_gsr)
        
        # store overall performance
        performance[name]['overall'].update({
            'avg_iou': avg_iou,
            'avg_normalized_precision_score': avg_nps,
            'avg_generalized_success_robustness_score': avg_gsr})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        #if not realtime:
        #    self.plot_curves(tracker_names)

        return performance


    def _calc_metrics(self, boxes, anno):
        ious = []
        norm_center_errors = [] 

        for box, a in zip(boxes, anno):
            if a[0] < 0 and a[1] < 0 and a[2] < 0 and a[3] < 0:
                continue
            else:
                ious.append(rect_iou(np.array([box]), np.array([a]))[0])
                norm_center_errors.append(normalized_center_error(np.array([box]), np.array([a]))[0])

        ious = np.array(ious)
        norm_center_errors = np.array(norm_center_errors)
        
        return ious, norm_center_errors

    def _calc_metrics_segm(self, segm, anno_segm):
 
        ious = segm_iou(segm, anno_segm)
        norm_center_errors = normalized_center_error_segm(segm, anno_segm)
        
        return ious, norm_center_errors

    def _calc_curves(self, ious, norm_center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        norm_center_errors = np.asarray(norm_center_errors, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_nce = np.linspace(0, 0.5, self.nbins_nce)[np.newaxis, :]

        bin_iou = np.greater(ious, thr_iou)
        bin_nce = np.less_equal(norm_center_errors, thr_nce)

        succ_curve = np.mean(bin_iou, axis=0)
        norm_prec_curve = np.mean(bin_nce, axis=0)

        return succ_curve, norm_prec_curve

    def _calc_curves_robustness(self, ious):
        seq_length = ious.shape[0]

        thr_iou = np.linspace(0, 0.5, self.nbins_gsr)

        gen_succ_rob_curve = np.zeros(thr_iou.shape[0])
        for i, th in enumerate(thr_iou):
            broken = False
            for j, iou in enumerate(ious):
                if iou <= th:
                    gen_succ_rob_curve[i] = float(j) / seq_length
                    broken = True
                    break
            if not broken:
                gen_succ_rob_curve[i] = 1.0

        return gen_succ_rob_curve

    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))
        
        play_speed = int(round(play_speed))
        assert play_speed > 0
        self.dataset.return_meta = False

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))
            
            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, seq_name,
                    '%s_001.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')
            
            # loop over the sequence and display results
            img_files, anno = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_dir, boxes, times):
        # record bounding boxes
        
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)

        record_file_boxes = os.path.join(record_dir, 'boxes.txt')
        np.savetxt(record_file_boxes,boxes, fmt='%.3f', delimiter=',')
        while not os.path.exists(record_file_boxes):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file_boxes, boxes, fmt='%.3f', delimiter=',')
        print('  Box results recorded at', record_file_boxes)

        #record_file_confs = os.path.join(record_dir, 'confidences.txt')
        #np.savetxt(record_file_confs, np.array([]), fmt='%.8f', delimiter='\n')
        #while not os.path.exists(record_file_confs):
         #   print('warning: recording failed, retrying...')
          #  np.savetxt(record_file_confs, fmt='%.8f', delimiter='\n')
        #print('  Conf results recorded at', record_file_confs)

        # record running times
        record_file_times = os.path.join(record_dir, 'times.txt')
        np.savetxt(record_file_times,times, fmt='%.8f', delimiter='\n')
        while not os.path.exists(record_file_times):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file_times, times, fmt='%.8f', delimiter='\n')
        print('  Time results recorded at', record_file_times)

    def _check_deterministic(self, tracker_name, seq_name):
        record_dir = os.path.join(
            self.result_dir, tracker_name, seq_name)
        record_files = sorted(glob.glob(os.path.join(
            record_dir, '%s_[0-9]*.txt' % seq_name)))

        if len(record_files) < 3:
            return False

        records = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                records.append(f.read())
        
        return len(set(records)) == 1

    def _evaluate(self, ious, times):
        # AO, SR and tracking speed
        ao = np.mean(ious)
        sr = np.mean(ious > 0.5)
        if len(times) > 0:
            # times has to be an array of positive values
            speed_fps = np.mean(1. / times)
        else:
            speed_fps = -1

        # success curve
        # thr_iou = np.linspace(0, 1, 101)
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        bin_iou = np.greater(ious[:, None], thr_iou[None, :])
        succ_curve = np.mean(bin_iou, axis=0)

        return ao, sr, speed_fps, succ_curve

    def plot_curves(self, report_files, tracker_names, extension='.png'):
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)
        
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        succ_file = os.path.join(report_dir, 'success_plot'+extension)
        key = 'overall'
        
        # filter performance by tracker_names
        performance = {k:v for k,v in performance.items() if k in tracker_names}

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t[key]['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]
        
        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['succ_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (
                name, performance[name][key]['ao']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower left',
                           bbox_to_anchor=(0., 0.))
        
        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on GOT-10k')
        ax.grid(True)
        fig.tight_layout()
        
        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)