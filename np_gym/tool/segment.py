
from gym_np import model
from gym_np.tool.common import filter
from pathlib import Path
import numpy as np
import time
import cv2

class SegmentEngine():
    def __init__(self, 
                    segment_net_file=None, 
                    process_type='segment_script', 
                    robot_type="ambf", 
                    image_type="zoom_needle_gripper_boximage",
                    is_save_anomaly_pic=True,
                    anomaly_pic_path=None,
                    zoom_margin_ratio=0.3,
                    ):
        assert process_type in ['segment_script','segment_net'], process_type
        self.process_type = process_type
        self.robot_type =robot_type
        self.image_type = image_type
        self.is_save_anomaly_pic = is_save_anomaly_pic
        self.anomaly_pic_path =anomaly_pic_path or str(Path((model.__path__)[0]).parent.parent / 'data' / 'seg' / 'anomaly')
        self.current_time = time.time()
        self.zoom_margin_ratio = zoom_margin_ratio
        if process_type == 'segment_net':
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
            print("init segment engine.....")
            # self.path = load_path or (model.__path__)[0]
            # self.path = Path(self.path)
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = segment_net_file or str(Path((model.__path__)[0]).parent / 'model' / 'segment.pth')
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # if 
            self.cfg.MODEL.DEVICE='cuda'
            print(f"Segment_net model file: {self.cfg.MODEL.WEIGHTS}")
            self.predictor = DefaultPredictor(self.cfg)
            print("finish")
    
    def predict_mask(self, rgb_arr):
        if self.process_type == 'segment_script':
            return self._predict_mask_script(rgb_arr)
        elif self.process_type == 'segment_net':
            return self._predict_mask_net(rgb_arr)
    
    def _predict_mask_script(self, rgb_arr):
        if self.robot_type == "ambf":
            # needle_bools=np.full(rgb_arr[:,:,0].shape, False, dtype=bool)
            # needle_bools=np.full(rgb_arr[:,:,0].shape, True, dtype=bool)
            # needle_bools = self._mask_filter_value_compare(rgb_arr,  c1=0, c2=1, bools=needle_bools, gap_up_bd=115)
            # needle_bools = self._mask_filter_value(rgb_arr, bools=needle_bools,channel_idx=0,low_bd=None, up_bd=150)
            # needle_bools = self._mask_filter_value(rgb_arr, bools=needle_bools,channel_idx=1,low_bd=4, up_bd=230)
            # needle_bools = self._mask_filter_value(rgb_arr, bools=needle_bools,channel_idx=2,low_bd=None, up_bd=170)


            # # gripper_bools =np.full(rgb_arr[:,:,0].shape, False, dtype=bool)
            # gripper_bools =np.full(rgb_arr[:,:,0].shape, True, dtype=bool)
            # gripper_bools = self._mask_filter_value(rgb_arr, bools=gripper_bools,channel_idx=2,low_bd=100, up_bd=None)
            # gripper_bools = self._mask_filter_value(rgb_arr, bools=gripper_bools, channel_idx=0,low_bd=None, up_bd=40)
            
            # needle_bools =np.full(rgb_arr[:,:,0].shape, False, dtype=bool)
            needle_bools=np.full(rgb_arr[:,:,0].shape, True, dtype=bool)
            # needle_bools = self._mask_filter_value_compare(rgb_arr,  c1=1, c2=0, bools=needle_bools, gap_low_bd=1)
            needle_bools = self._mask_filter_value_compare(rgb_arr,  c1=1, c2=2, bools=needle_bools, gap_low_bd=5)
            # needle_bools = self._mask_filter_value(rgb_arr, bools=needle_bools,channel_idx=1,low_bd=50, up_bd=120)



            gripper_bools =np.full(rgb_arr[:,:,0].shape, False, dtype=bool)
            gripper_bools =np.full(rgb_arr[:,:,0].shape, True, dtype=bool)
            gripper_bools = self._mask_filter_value_compare(rgb_arr,  c1=2, c2=0, bools=gripper_bools, gap_low_bd=20)
            gripper_bools = self._mask_filter_value_compare(rgb_arr,  c1=2, c2=1, bools=gripper_bools, gap_low_bd=20)
            # gripper_bools = self._mask_filter_value(rgb_arr, bools=gripper_bools,channel_idx=2,low_bd=None, up_bd=115)
            
        elif self.robot_type == "dvrk":
            # needle_bools =np.full(rgb_arr[:,:,0].shape, False, dtype=bool)
            needle_bools=np.full(rgb_arr[:,:,0].shape, True, dtype=bool)
            needle_bools = self._mask_filter_value_compare(rgb_arr,  c1=1, c2=0, bools=needle_bools, gap_low_bd=35)
            needle_bools = self._mask_filter_value_compare(rgb_arr,  c1=1, c2=2, bools=needle_bools, gap_low_bd=-6)



            # gripper_bools =np.full(rgb_arr[:,:,0].shape, False, dtype=bool)
            gripper_bools =np.full(rgb_arr[:,:,0].shape, True, dtype=bool)
            gripper_bools = self._mask_filter_value_compare(rgb_arr,  c1=0, c2=1, bools=gripper_bools, gap_low_bd=67)
            gripper_bools = self._mask_filter_value_compare(rgb_arr,  c1=0, c2=2, bools=gripper_bools, gap_low_bd=59)
      
        if (not np.any(needle_bools)) or (not np.any(gripper_bools)):
            return None, False
        
        
        # print(needle_bools.shape)
        result = {}
        result[int(0)] = (gripper_bools, 1)
        result[int(1)] = (needle_bools, 1)
    
        return result, True

    def _predict_mask_net(self, rgb_arr):
        to_np = lambda x: x.detach().cpu().numpy()
        # start = time.time()
        result = self.predictor(rgb_arr)
        # print(result['instances'].pred_classes)
        # print(f"predict elapse time {time.time()-start}")
        masks = {}
        

        for j in range(result['instances'].pred_classes.shape[0]):
            _class_id = int(to_np(result['instances'].pred_classes[j]))
            # print(_class_id)
            # print(masks)
            if _class_id in masks:
                # if masks[_class_id][1] < to_np(result['instances'].scores[j]):
                masks[_class_id] = (to_np(result['instances'].pred_masks[j]) | masks[_class_id][0], 
                                    min(to_np(result['instances'].scores[j]), masks[_class_id][1]),
                                    )
            else:
                masks[_class_id] = (to_np(result['instances'].pred_masks[j]), 
                                    to_np(result['instances'].scores[j]))      
        # out = result['instances'].pred_masks.detach().cpu().numpy()
        # masks = [out[i,:,:] for i in range(out.shape[0])]
        # is_sucess = len(masks.keys())  ==2 # stop if score is less than set threshold, need more label data for training
        # print(masks)
        if len(masks.keys())==2:
            return masks, True 
        else: # fail
            if self.is_save_anomaly_pic and float(time.time()-self.current_time) > 0.5:
                self.current_time = time.time()
                _anomaly_pic_path = Path(self.anomaly_pic_path) / self.robot_type 
                _anomaly_pic_path.mkdir(parents=True, exist_ok=True)
                _file =  "anomaly_" + time.strftime("%Y%m%d-%H%M%S") + ".png"
                cv2.imwrite(str(_anomaly_pic_path / _file), cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
            return None, False
    
    def segment(self, matrix, mask):
        mask_matrix = np.zeros(matrix.shape, dtype=np.uint8)
        # print(matrix)
        # print(mask.shape)
        mask_matrix[mask] = matrix[mask]
        # print(np.sum(mask_matrix))
        return mask_matrix
    
    def segment_image(self, segment_input_im,  render_im, segment_object="all"):
        assert segment_input_im.shape == render_im.shape
        assert segment_input_im.shape[2] == 3
        masks, is_sucess = self.predict_mask(segment_input_im)
        if not is_sucess:
            raise NotImplementedError()
            return render_im
        

        if segment_object=="all":
            mask = masks[1][0] | masks[0][0]
        elif segment_object=="gripper":
            mask = masks[0][0]
        elif segment_object=="needle":
            mask = masks[1][0]

        render_ims =[]
        for i in range(3):
            render_ims.append(self.segment(render_im[:,:,i], mask))

        render_ims = np.stack(render_ims, axis=2)    
        return render_ims
        
    
    def default_result_stat(self):
        return {"image":None, 
                "is_success": False,
                   "gripper_box_x_rel":(0.0, 0.0),
                   "gripper_box_y_rel":(0.0, 0.0),
                   "needle_box_x_rel": (0.0, 0.0),
                   "needle_box_y_rel": (0.0, 0.0),
                   "gripper_box_center_pos": (0.0, 0.0, 0.0), 
                   "needle_box_center_pos": (0.0, 0.0, 0.0), 
                   }  # default


    def process_image(self, im, depth=None):
        results = self.default_result_stat()
        results["image"] = im
        masks, is_sucess = self.predict_mask(im)
        if not is_sucess:
            results["image"] = np.zeros(im.shape, dtype=np.uint8)
            results["is_success"] = False
            return results
        
        if self.image_type == "origin_rgb":
            im_pre = im
            results["image"] = im_pre
            results["is_success"] = True
            return results
        elif self.image_type == "origin_depth":
            im_pre = np.stack([depth[:, :, 0]]*3, axis=2)
            results["image"] = im_pre
            results["is_success"] = True
            return results
        elif self.image_type == "origin_mix":
            _stacks = []
            _stacks.append(
                (0.1*im[:, :, 0] + 0.9*im[:, :, 1]).astype(np.uint8))
            _stacks.append(im[:, :, 2])
            _stacks.append(depth[:, :, 0])
            im_pre = np.stack(_stacks, axis=2)
            results["image"] = im_pre
            results["is_success"] = True
            return results
        elif self.image_type == "zero":
            results["image"] = np.zeros(im.shape, dtype=np.uint8)
            results["is_success"] = True
            return results
        elif self.image_type == "mask_full":
            im_pre = np.zeros(im.shape, dtype=np.uint8)
            im_pre[masks[0][0],2] = 255
            im_pre[masks[1][0],1] = 255
            results["image"] = im_pre
            results["is_success"] = True
            return results           
        elif self.image_type == "mask_depth":
            im_pre = np.zeros(im.shape, dtype=np.uint8)
            im_pre[masks[0][0],2] = depth[masks[0][0],0]
            im_pre[masks[1][0],1] = depth[masks[1][0],0]
            results["image"] = im_pre
            results["is_success"] = True
            return results 
        
        results['needle_x_mean'], results['needle_y_mean'] = self.get_x_y_mean_from_masks(masks[1][0])
        results['needle_area'] = self.get_area_from_mask(masks[1][0])
        # print(results['needle_area'])
        _out = self.get_box_from_masks(
            masks[1][0])
        if _out is None:
            return results
        else:         
            results['needle_box_x'], results['needle_box_y'], results['needle_box_x_rel'], results['needle_box_y_rel'] = _out 

        results['gripper_x_mean'], results['gripper_y_mean'] = self.get_x_y_mean_from_masks(masks[0][0])
        results['gripper_area'] = self.get_area_from_mask(masks[0][0])
        _out = self.get_box_from_masks(
            masks[0][0])
        if _out is None:
            return results
        else:         
            results['gripper_box_x'], results['gripper_box_y'], results['gripper_box_x_rel'], results['gripper_box_y_rel'] = _out 


        if depth is None:
            # mask = (masks[0][0] | masks[1][0])
            # r_mat = self.segment(im[:,:,0], mask)
            # g_mat = self.segment(im[:,:,1], mask)
            # b_mat = self.segment(im[:,:,2], mask)
            # im_pre = np.stack([r_mat,g_mat,b_mat], axis=2)
            raise NotImplementedError
            
        else:
            gripper_mat = self.segment(depth[:,:,0], masks[0][0])
            needle_mat = self.segment(depth[:,:,0], masks[1][0])
            results['gripper_value_mean'] = np.mean(depth[:,:,0][masks[0][0]]) / 255.0
            results['needle_value_mean'] = np.mean(depth[:,:,0][masks[1][0]]) / 255.0
            mean_fuc = lambda x: sum(list(x)) / len(list(x)) 
            get_center_pos = lambda x,y,z: (mean_fuc(x),mean_fuc(y),z,)
            results['gripper_box_center_pos'] = get_center_pos(results['gripper_box_x_rel'],results['gripper_box_y_rel'],results['gripper_value_mean'])
            results['needle_box_center_pos'] = get_center_pos(results['needle_box_x_rel'],results['needle_box_y_rel'],results['needle_value_mean'])
            if self.image_type == "depth_seg_normal":
                background_mat = np.full(gripper_mat.shape, 0,dtype=np.uint8)
                segs = [gripper_mat, needle_mat, background_mat]
                im_pre = np.stack(segs, axis=2)
            elif self.image_type == "depth_seg_enhance":
                depth_mat = self.segment(depth[:, :, 0], masks[0][0]|masks[1][0])
                gripper_enhance_mat = self.segment(np.full(depth_mat.shape, 255,dtype=np.uint8), masks[0][0])
                needle_enhance_mat = self.segment(np.full(depth_mat.shape, 255,dtype=np.uint8), masks[1][0])
                segs = [depth_mat, needle_enhance_mat, gripper_enhance_mat]
                im_pre = np.stack(segs, axis=2)
                
            elif self.image_type == "zoom_needle_gripper_boximage" or self.image_type == "zoom_needle_gripper_boxscalar":
                depth_mat = depth[:, :, 0]
                _target_shape = depth_mat.shape
                needle_zoom_mat, _x_needle, _y_needle = self._get_zoom_mat(
                    masks[1][0], depth_mat, results['needle_box_x'], results['needle_box_y'], )
                gripper_zoom_mat, _x_gripper, _y_gripper = self._get_zoom_mat(
                    masks[0][0], depth_mat, results['gripper_box_x'], results['gripper_box_y'], )
                if self.image_type == "zoom_needle_gripper_boximage":
                    gripper_box_mask = np.full(depth_mat.shape, 0, dtype=np.uint8)
                    gripper_box_mask[_x_gripper[0]:_x_gripper[1],
                                    _y_gripper[0]:_y_gripper[1]] = 200
                
                    needle_box_mask = np.full(depth_mat.shape, 0, dtype=np.uint8)
                    needle_box_mask[_x_needle[0]:_x_needle[1],
                                    _y_needle[0]:_y_needle[1]] = 50
                    
                    box_mat = gripper_box_mask + needle_box_mask
                else:
                    box_mat = np.full(depth_mat.shape, 0, dtype=np.uint8)
                segs = [box_mat, needle_zoom_mat, gripper_zoom_mat]
                im_pre = np.stack(segs, axis=2)
            
            elif self.image_type == "zoom_needle_boximage" or self.image_type == "zoom_needle_boxscalar":
                depth_mat = depth[:, :, 0]
                # gripper_mat = self.segment(depth_mat, masks[0][0])
                _zoom_needle_mat, _x_needle, _y_needle = self._get_zoom_mat(
                    masks[1][0] , depth_mat, results['needle_box_x'], results['needle_box_y'], margin_ratio=self.zoom_margin_ratio)
                _zoom_gripper_mat, _, _ = self._get_zoom_mat(
                    masks[0][0], depth_mat, results['needle_box_x'], results['needle_box_y'], margin_ratio=self.zoom_margin_ratio)
                if self.image_type == "zoom_needle_boximage":
                    gripper_mat[_x_needle[0]:_x_needle[1],
                                    _y_needle[0]:_y_needle[1]] = 255
                else:
                    gripper_mat = np.zeros(depth[:, :, 0].shape, dtype=np.uint8)
                segs = [gripper_mat, _zoom_needle_mat, _zoom_gripper_mat]
                im_pre = np.stack(segs, axis=2)
            elif  self.image_type == "zoom_mask_boximage":
                mask_mat = np.full(im.shape[:2],255,np.uint8)
                _zoom_needle_mat, _x_needle, _y_needle = self._get_zoom_mat(
                    masks[1][0] , mask_mat, results['needle_box_x'], results['needle_box_y'], margin_ratio=self.zoom_margin_ratio)
                _zoom_gripper_mat, _, _ = self._get_zoom_mat(
                    masks[0][0], mask_mat, results['needle_box_x'], results['needle_box_y'], margin_ratio=self.zoom_margin_ratio)
            
                gripper_mat[_x_needle[0]:_x_needle[1],
                            _y_needle[0]:_y_needle[1]] = 255
                segs = [gripper_mat, _zoom_needle_mat, _zoom_gripper_mat]         
                im_pre = np.stack(segs, axis=2)   
            elif self.image_type == "origin_rgb":
                im_pre = im
            elif self.image_type == "origin_depth":
                im_pre = np.stack([depth[:, :, 0]]*3, axis=2)
            else:
                raise NotImplementedError
            
        results["image"] = im_pre
        results["is_success"] = True

        return results

    def _get_zoom_mat(self, mask_mat,depth_mat, box_x, box_y, margin_ratio=0.1, fix_box_ratio=None):
        zoom_mat = self.segment(
                    depth_mat, mask_mat)
                
        _zoom_box_length = max((box_x[1] - box_x[0]),
                                (box_y[1] - box_y[0])) * (1+margin_ratio)
        _zoom_box_cen_x = int((box_x[1] + box_x[0])//2)
        _zoom_box_cen_y = int((box_y[1] + box_y[0])//2)
        if fix_box_ratio is None:
            _l = int(_zoom_box_length // 2)
        else:
            _l = int((depth_mat.shape[0] * fix_box_ratio)//2)
        x_min = np.clip(_zoom_box_cen_x-_l, 0, depth_mat.shape[0]-1)
        x_max = np.clip(_zoom_box_cen_x+_l+1, 0, depth_mat.shape[0]-1)
        y_min = np.clip(_zoom_box_cen_y-_l, 0, depth_mat.shape[0]-1)
        y_max = np.clip(_zoom_box_cen_y+_l+1, 0, depth_mat.shape[0]-1)
        zoom_mat = zoom_mat[x_min:x_max,
                            y_min:y_max]
        
        try:
            zoom_mat = cv2.resize(zoom_mat,
                                        dsize=depth_mat.shape, interpolation=cv2.INTER_NEAREST)
        except:
            zoom_mat = np.full(
                depth_mat.shape, 0, dtype=np.uint8)
        return zoom_mat, (x_min, x_max), (y_min, y_max) 

    def get_area_from_mask(self, mask):
        return np.sum(mask)/ (mask.shape[0] * mask.shape[1])

    def get_x_y_mean_from_masks(self, mask):
        x_mean, y_mean = np.where(mask)
        x_mean = np.mean(x_mean)
        y_mean = np.mean(y_mean)
        _s = mask.shape
        return x_mean/_s[0], y_mean/_s[1]
    
    def get_box_from_masks(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        try:
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            _s = mask.shape
            return (rmin, rmax), (cmin, cmax), (rmin/_s[0], rmax/_s[0]), (cmin/_s[1], cmax/_s[1])
        except:
            return None

    def view(self, im, im2=None, alpha=1, beta=0.1):
        if im2 is not None:
            _im = cv2.addWeighted(im, alpha, im2, beta, 0.0)
        else:
            _im = im
        frame = cv2.resize(_im, (1080, 1080), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('preview', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # Display the resulting frame
        k = cv2.waitKey(0)
        
    
    def _mask_filter_value(self, im,channel_idx, bools=None, low_bd=None, up_bd=None):
        im_filtered = im
        if bools is None:
            bools = np.full(im[:,:,0].shape, True, dtype=bool)
        if low_bd is not None:
            bools = (im_filtered[:,:,channel_idx] >= low_bd) & bools
        if up_bd is not None:
            bools = (im_filtered[:,:,channel_idx] <= up_bd) & bools
        return bools


    def _mask_filter_value_compare(self, im, c1, c2, bools=None, gap_low_bd=None, gap_up_bd=None):
        im_filtered = im.astype(np.int32)
        # print(im.shape)
        gap_px = im_filtered[:,:,c1] - im_filtered[:,:,c2]
        if bools is None:
            bools = np.full(im[:,:,0].shape, True, dtype=bool)
        if gap_low_bd is not None:
            bools = (gap_px >= gap_low_bd) & bools
        if gap_up_bd is not None:
            bools = (gap_px <= gap_up_bd) & bools
        return bools

            
            
if __name__ == '__main__':
    import cv2
    import sys
    import argparse
    from gym_np.env.wrapper import make_env
    from time import sleep
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--method', type=str, default='script')
    parser.add_argument('--robot', type=str, default='ambf')
    parser.add_argument('--beta', type=float, default=0.1)
    args = parser.parse_args()

    engine = SegmentEngine(robot_type=args.robot)
    im = cv2.imread(args.input)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_proc, is_sucess = engine.process_image(im, depth=None, is_gripper_close=True)
    if not is_sucess:
        print("segment fail")
        sys.exit()
    engine.view(im_proc,im2=im, beta=args.beta)
