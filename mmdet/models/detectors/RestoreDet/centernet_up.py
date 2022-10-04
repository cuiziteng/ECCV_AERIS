import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import torchvision.transforms.functional as F
from mmcv.image import tensor2imgs
import matplotlib.pyplot as plt

# kornia: a GPU-based torch library, here we use it to adopt gaussian blur on tensor
# source paper: 'https://arxiv.org/pdf/1910.02190.pdf'
from kornia.filters import gaussian_blur2d, motion_blur, filter2D
import random
import numpy as np

from .kernel import isotropic_Gaussian, anisotropic_Gaussian

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS, build_backbone

from ....core.utils import flip_tensor
from ..single_stage import SingleStageDetector


@DETECTORS.register_module()
class CenterNet_Up(SingleStageDetector):
    """Implementation of CenterNet(Objects as Points)

    <https://arxiv.org/abs/1904.07850>.
    """

    def __init__(self,
                 backbone,
                 neck,
                 arrd,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CenterNet_Up, self).__init__(backbone, neck, bbox_head, train_cfg,
                                           test_cfg, pretrained, init_cfg)
        self.ARRD = build_backbone(arrd)
        self.sr_loss = nn.L1Loss()


    def random_deg(self, img, img_meta, l_size, gt_bbox, keep_shape=False):
        '''
        This Part is based on GPU, because too much CPU consume in Dataloader part!!!
        input:
        img (Tensor): Input images of shape (C, H, W).
        img_meta (dict): A image info dict contain some information like name ,shape ...
        keep_shape (bool): Choose to return same size or lower size
        s (float): down-sampling scale factor ,from 1~4
        gt_bbox (Tensor): original bbox label of image

        return:
        keep_shape = True: img_low (Tensor): Output degraded images of shape (C, H, W).
        keep_shape = False: img_low (Tensor): Output degraded images of shape (C, H/ratio, W/ratio).
        '''
        device = img.device

        # here use unsqueeze, this is because torch's bicubic can only implenment on 4D tenspr
        img_hr = img.unsqueeze(0)  # (1, C, H, W)
        kernel_size = int(random.choice([7, 9, 11, 13, 15, 17, 19, 21]))
        kernel_width_iso = random.uniform(0.1, 2.4)
        angle = random.uniform(0, np.pi)
        kernel_width_un1 = random.uniform(0.5, 6)
        kernel_width_un2 = random.uniform(0.5, kernel_width_un1)

        # random choose from three degradation types
        deg_type = random.choice(['none', 'iso', 'aniso'])
        # random choose from three interpolate types
        scale_mode = random.choice(['nearest', 'bilinear', 'bicubic'])

        # Choose to use or not use blur
        if deg_type == 'none':  # adopt none blur
            img_blur = img_hr

        elif deg_type == 'iso':  # adopt isotropic Gaussian blur
            k = isotropic_Gaussian(kernel_size, kernel_width_iso)
            k_ts = torch.from_numpy(k).to(torch.device(device)).unsqueeze(0)
            img_blur = filter2D(img_hr, k_ts)

        elif deg_type == 'aniso':  # adopt anisotropic Gaussian blur
            k = anisotropic_Gaussian(kernel_size, angle, kernel_width_un1, kernel_width_un2)
            k_ts = torch.from_numpy(k).to(torch.device(device)).unsqueeze(0)
            img_blur = filter2D(img_hr, k_ts)


        # Down Sampling, rate random choose from 1~4

        img_dr = interpolate(img_blur, size=(l_size, l_size), mode=scale_mode)
        s = 512/l_size
        # add noise
        noise_level_img = random.uniform(0, 25) / 255.0
        noise = torch.normal(mean=0.0, std=noise_level_img, size=img_dr.shape).to(torch.device(device))
        img_dr += noise

        if not keep_shape:
            gt_bbox_down = gt_bbox/s
            new_h, new_w = int(img_meta['pad_shape'][0]/s), int(img_meta['pad_shape'][1]/s)
            img_meta['pad_shape'] = (new_h, new_w, 3)
            return img_dr.squeeze(0), gt_bbox_down

        # keep the same shape as img_hr, default = False
        else:
            # bicubic up-scale
            img_lr = interpolate(img_dr, (img_blur.shape[2], img_blur.shape[3]), mode='bicubic')
            return img_lr.squeeze(0)

    def extract_feat(self, img_lr):
        """Directly extract features from the backbone+neck."""
        x_l = list(self.backbone(img_lr))
       
        if self.with_neck:
            x_l = self.neck(x_l)
        #print(x_l[-1].shape)
        img_sr = self.ARRD(x_l[-1])
        #print('111', img_sr.shape)
        return x_l, img_sr
    
    def extract_feat_test(self, img_lr):
        """Directly extract features from the backbone+neck."""
        x_l = list(self.backbone(img_lr))
       
        if self.with_neck:
            x_l = self.neck(x_l)
        
        return x_l

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        #if random.random() > 0.5:
        #super(CenterNet_UP, self).forward_train(img, img_metas, gt_bboxes, gt_labels)
        batch_size = img.shape[0]
        device = img.device
        # down sample rate random choose from 1~4
        #print(img.shape)
        scale_list = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        l_size = random.choice(scale_list) * 32 # Must be Multiples of 32, 128 -- 512

        img_lr = torch.empty(size=(batch_size, 3, l_size, l_size)).to(torch.device(device))
        #print(img_lr.shape)
        gt_bboxes_down = []
        
        for i in range(batch_size):
            img_lr[i], gt_bbox_down = self.random_deg(img[i], img_metas[i], l_size, gt_bboxes[i])
            gt_bboxes_down.append(gt_bbox_down)

        
        #x, img_sr = self.extract_feat(img_lr, 512/l_size)
        x, img_sr = self.extract_feat(img_lr)

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes_down,
                                                gt_labels, gt_bboxes_ignore)
        losses['sr_loss'] = 0.2 * self.sr_loss(img_sr, img)
        
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat_test(img)

        results_list= self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)

        #heat_map = heat_map.squeeze(0)
        #heat_map_img = heat_map.cpu().numpy().squeeze(axis=0)
        #print('22222',heat_map_img.shape)
        #heat_map_img = np.stack((heat_map_img,heat_map_img,heat_map_img),axis=-1)
        #print(heat_map_img.shape)
        #print(img_metas[0]['ori_filename'])

        #SAVE_PATH = r'/home/czt/SHOW_dir/HEAT/' + img_metas[0]['ori_filename']
        #scipy.misc.toimage(heat_map_img).save(SAVE_PATH)
        #plt.imsave(SAVE_PATH, heat_map_img, cmap='gray')
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]

        return bbox_results

    def merge_aug_results(self, aug_results, with_nms):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            with_nms (bool): If True, do nms before return boxes.

        Returns:
            tuple: (out_bboxes, out_labels)
        """
        recovered_bboxes, aug_labels = [], []
        for single_result in aug_results:
            recovered_bboxes.append(single_result[0][0])
            aug_labels.append(single_result[0][1])

        bboxes = torch.cat(recovered_bboxes, dim=0).contiguous()
        labels = torch.cat(aug_labels).contiguous()
        if with_nms:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(
                bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = bboxes, labels

        return out_bboxes, out_labels

    def aug_test(self, imgs, img_metas, rescale=True):
        """Augment testing of CenterNet. Aug test must have flipped image pair,
        and unlike CornerNet, it will perform an averaging operation on the
        feature map instead of detecting bbox.

        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.

        Note:
            ``imgs`` must including flipped image pairs.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img_inds = list(range(len(imgs)))
        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
            'aug test must have flipped image pair')
        aug_results = []
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            flip_direction = img_metas[flip_ind][0]['flip_direction']
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            center_heatmap_preds, wh_preds, offset_preds = self.bbox_head(x)
            assert len(center_heatmap_preds) == len(wh_preds) == len(
                offset_preds) == 1

            # Feature map averaging
            center_heatmap_preds[0] = (center_heatmap_preds[0][0:1] +
                                              flip_tensor(center_heatmap_preds[0][1:2], flip_direction)) / 2
            wh_preds[0] = (wh_preds[0][0:1] +
                           flip_tensor(wh_preds[0][1:2], flip_direction)) / 2

            bbox_list = self.bbox_head.get_bboxes(
                center_heatmap_preds,
                wh_preds, [offset_preds[0][0:1]],
                img_metas[ind],
                rescale=rescale,
                with_nms=False)
            aug_results.append(bbox_list)

        nms_cfg = self.bbox_head.test_cfg.get('nms_cfg', None)
        if nms_cfg is None:
            with_nms = False
        else:
            with_nms = True
        bbox_list = [self.merge_aug_results(aug_results, with_nms)]
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
