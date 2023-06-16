import numpy as np
from gym_suture.param.default import default
from copy import deepcopy

#==========ambf phantom===============
ambf_phantom_param = deepcopy(default)

ambf_phantom_param.NeedlePicking_wrapper.update({
"camL_local_pos": [0.1867,0.7005,-0.7930],
"camL_local_quat": [ 0.0616,-0.0616,0.7044,0.7044]
})



ambf_phantom_param.ImagePreprocess_wrapper.update({
    "zoom_margin_ratio": 0.5,
}
)