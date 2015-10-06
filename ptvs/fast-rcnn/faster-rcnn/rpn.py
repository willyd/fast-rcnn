"""
This module implements the region proposal network RPN described in
Faster R-CNN
"""
import numpy as np
import caffe

def _ratio_jitter(anchors, ratios):
    """
    Takes input anchors and create
    anchors based on the ratios input.
    """
    assert len(anchors.shape) == 2 and anchors.shape[1] == 4
    output_anchors = []
    for anchor in anchors:
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + (w - 1) / 2
        y_ctr = anchor[1] + (h - 1) / 2

        size = w * h
    
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
    
        output_anchors.append(np.hstack((x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, 
                             x_ctr + (ws - 1) / 2, y_ctr + (hs - 1) / 2)))

    return np.vstack(output_anchors)

def _scale_jitter(anchors, scales):
    """
    Takes input anchors and create
    anchors based on the scales input.
    """
    assert len(anchors.shape) == 2 and anchors.shape[1] == 4
    output_anchors = []
    for anchor in anchors:
        w = (anchor[2] - anchor[0] + 1)
        h = (anchor[3] - anchor[1] + 1)
        x_ctr = (anchor[0] + (w - 1) / 2)
        y_ctr = (anchor[1] + (h - 1) / 2)

        ws = w * scales
        hs = h * scales
    
        output_anchors.append(np.hstack((x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, 
                                         x_ctr + (ws - 1) / 2, y_ctr + (hs - 1) / 2)))

    return np.vstack(output_anchors)

def generate_proposal_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate an array of N x 4 anchors for RPN 
    """
    if ratios is None:
        ratios = np.array([0.5, 1.0, 2.0]).reshape(-1, 1)
    if scales is None:
        scales = np.power(2, np.array(range(3,6))).reshape(-1, 1)

    base_anchor = np.array([0, 0, base_size-1, base_size-1], dtype=float).reshape(1, -1)
    ratio_anchors = _ratio_jitter(base_anchor, ratios)
    anchors = _scale_jitter(ratio_anchors, scales)

    return anchors

def proposal_compute_output_size(net, min_size=100, max_size=1000):
    """
    Generate two dicts for size to ouput score size.
    net is a TEST net. Should probably compute the size without 
    doing a forward pass in the net.
    """
    assert isinstance(net, caffe.Net)

    input_size = range(min_size, max_size+1)
    output_width  = {}
    output_height = {}
    blobs_in = dict([(name, net.blobs[name].data) for name in net.inputs])
    for s in input_size:
        im_blob = np.zeros((1, 3, s, s), dtype=np.float32)        
        blobs_in['data'] = im_blob
        net.blobs['data'].reshape(*(im_blob.shape))
        _ = net.forward(**blobs_in)
        cls_score = net.blobs['proposal_cls_score']
        output_width[s] = cls_score.shape[2]
        output_height[s] = cls_score.shape[3]

    return output_width, output_height

def main():
    pass

if __name__ == '__main__':
    #main()
    #a = generate_proposal_anchors()
    caffe.set_mode_gpu()

    filename = r"C:\Work\src\aware2\aware2\faster_rcnn\models\rpn_prototxts\ZF\test.prototxt"
    net = caffe.Net(filename, caffe.TEST)
    ow, oh = proposal_compute_output_size(net, 100, 100)
    