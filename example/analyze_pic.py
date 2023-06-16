#refer: https://blog.csdn.net/weixin_42216109/article/details/89520423

import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
args = parser.parse_args()


#定义窗口名称
winName='Colors of the rainbow'
#定义滑动条回调函数，此处pass用作占位语句保持程序结构的完整性
def nothing(x):
    pass
img_original=cv2.imread(args.dir)
#颜色空间的转换

def _mask_filter_value_compare(im, c1, c2, bools=None, gap_low_bd=None, gap_up_bd=None):
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
def _mask_filter_value(im,channel_idx, bools=None, low_bd=None, up_bd=None):
    im_filtered = im
    if bools is None:
        bools = np.full(im[:,:,0].shape, True, dtype=bool)
    if low_bd is not None:
        bools = (im_filtered[:,:,channel_idx] >= low_bd) & bools
    if up_bd is not None:
        bools = (im_filtered[:,:,channel_idx] <= up_bd) & bools
    return bools

img_rgb=cv2.cvtColor(img_original,cv2.COLOR_BGR2RGB)
#新建窗口
cv2.namedWindow(winName)

_color_range = 700
cv2.createTrackbar('Blue2Red_low',winName,0,_color_range,nothing)
cv2.createTrackbar('Blue2Red_high',winName,_color_range,_color_range,nothing)
cv2.createTrackbar('Blue2Green_low',winName,0,_color_range,nothing)
cv2.createTrackbar('Blue2Green_high',winName,_color_range,_color_range,nothing)
cv2.createTrackbar('Red2Green_low',winName,0,_color_range,nothing)
cv2.createTrackbar('Red2Green_high',winName,_color_range,_color_range,nothing)

cv2.createTrackbar('gray_low',winName,0,_color_range,nothing)
cv2.createTrackbar('gray_high',winName,_color_range,_color_range,nothing)


is_mask = False
while(1):
    
    Blue2Red_low=cv2.getTrackbarPos('Blue2Red_low',winName) - _color_range/2
    Blue2Red_high=cv2.getTrackbarPos('Blue2Red_high',winName) - _color_range/2
    masks = _mask_filter_value_compare(img_rgb,2,0, gap_low_bd=Blue2Red_low, gap_up_bd=Blue2Red_high)

    Blue2Green_low=cv2.getTrackbarPos('Blue2Green_low',winName) - _color_range/2
    Blue2Green_high=cv2.getTrackbarPos('Blue2Green_high',winName) - _color_range/2
    masks = _mask_filter_value_compare(img_rgb,2,1, bools=masks, gap_low_bd=Blue2Green_low, gap_up_bd=Blue2Green_high)

    Red2Green_low=cv2.getTrackbarPos('Red2Green_low',winName) - _color_range/2
    Red2Green_high=cv2.getTrackbarPos('Red2Green_high',winName) - _color_range/2
    masks = _mask_filter_value_compare(img_rgb,0,1, bools=masks, gap_low_bd=Red2Green_low, gap_up_bd=Red2Green_high)

    gray_low=cv2.getTrackbarPos('gray_low',winName) - _color_range/2
    gray_high=cv2.getTrackbarPos('gray_high',winName) - _color_range/2
    masks = _mask_filter_value(img_rgb, channel_idx=0, bools=masks, low_bd=gray_low, up_bd=gray_high)
    masks = _mask_filter_value(img_rgb, channel_idx=1, bools=masks, low_bd=gray_low, up_bd=gray_high)
    masks = _mask_filter_value(img_rgb, channel_idx=2, bools=masks, low_bd=gray_low, up_bd=gray_high)

    render_im = np.zeros(img_rgb.shape,dtype=np.uint8)
    _masks = np.stack([masks]*3, axis=2)
    render_im[_masks] = img_rgb[_masks] 
    cv2.imshow(winName, cv2.resize(cv2.cvtColor(render_im, cv2.COLOR_RGB2BGR), (500,500)))
    if cv2.waitKey(1)==ord('q'):
        break
    elif cv2.waitKey(1)==ord('p'):
        print(f"blue2red: low:{Blue2Red_low}, high:{Blue2Red_high}")
        print(f"blue2green: low:{Blue2Green_low}, high:{Blue2Green_high}")
        print(f"red2green: low:{Red2Green_low}, high:{Red2Green_high}")
    elif cv2.waitKey(1)==ord('m'):
        is_mask = not is_mask
        
cv2.destroyAllWindows()

