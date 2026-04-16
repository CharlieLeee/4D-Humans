import torch
import numpy as np
import pickle
from typing import Optional
import smplx
from smplx.lbs import vertices2joints
from smplx.utils import SMPLHOutput


class SMPL(smplx.SMPLHLayer):
    """
    SMPLH-based body model that accepts SMPL-format predictions (23 body joints).

    HMR2 was trained with SMPL (23 body joints). SMPLH has 21 body joints
    (the first 21 match SMPL exactly). This wrapper:
      - Accepts 23-joint body_pose from the pretrained HMR2 head
      - Passes the first 21 joints to SMPLHLayer
      - Discards SMPL joints 22-23 (L_Hand, R_Hand tips)
      - Sets hand poses to zero (flat hands)

    This allows using the widely available SMPLH_NEUTRAL.pkl model file
    with the existing HMR2 checkpoint without retraining.
    """

    # Number of SMPL body joints predicted by the HMR2 head
    NUM_SMPL_BODY_JOINTS = 23
    # Number of SMPLH body joints (excluding root)
    NUM_SMPLH_BODY_JOINTS = 21

    def __init__(self, *args, joint_regressor_extra: Optional[str] = None,
                 update_hips: bool = False, num_body_joints: int = 23, **kwargs):
        """
        Args:
            Same as SMPLHLayer, plus:
            joint_regressor_extra (str): Path to extra joint regressor.
            update_hips (bool): Whether to update hip joint positions.
            num_body_joints (int): Number of body joints from the HMR2 head
                (23 for SMPL-trained model). Kept for config compatibility.
        """
        # SMPLHLayer always uses 21 body joints internally; don't pass
        # the SMPL-specific num_body_joints to the parent.
        kwargs.pop('mean_params', None)  # not used by SMPLHLayer
        super().__init__(*args, use_pca=False, flat_hand_mean=True, **kwargs)

        # OpenPose-style joint mapping.
        # SMPLH has 52 joints (22 body + 30 hand). Extra regressed joints
        # start at index 52 (vs. index 24 in SMPL).
        smpl_to_openpose = [
            52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
            7, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62
        ]

        if joint_regressor_extra is not None:
            self.register_buffer(
                'joint_regressor_extra',
                torch.tensor(
                    pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'),
                    dtype=torch.float32,
                ),
            )
        self.register_buffer('joint_map', torch.tensor(smpl_to_openpose, dtype=torch.long))
        self.update_hips = update_hips

    def forward(self, *args, **kwargs) -> SMPLHOutput:
        """
        Run forward pass. Accepts SMPL-format body_pose (23 joints) and
        converts to SMPLH format (21 joints) before calling the parent.
        """
        body_pose = kwargs.get('body_pose', args[0] if args else None)
        pose2rot = kwargs.get('pose2rot', True)

        if body_pose is not None:
            # body_pose from HMR2: (B, 23, 3, 3) or (B, 69) for SMPL
            # We keep only the first 21 joints for SMPLH
            if pose2rot:
                # axis-angle: (B, 69) → (B, 63)
                if body_pose.shape[-1] == self.NUM_SMPL_BODY_JOINTS * 3:
                    body_pose = body_pose[..., :self.NUM_SMPLH_BODY_JOINTS * 3]
            else:
                # rotation matrices: (B, 23, 3, 3) → (B, 21, 3, 3)
                if body_pose.shape[-3] == self.NUM_SMPL_BODY_JOINTS:
                    body_pose = body_pose[:, :self.NUM_SMPLH_BODY_JOINTS, :, :]

            if 'body_pose' in kwargs:
                kwargs['body_pose'] = body_pose
            elif args:
                args = (body_pose,) + args[1:]

        smpl_output = super().forward(*args, **kwargs)
        joints = smpl_output.joints[:, self.joint_map, :]
        if self.update_hips:
            joints[:, [9, 12]] = (
                joints[:, [9, 12]]
                + 0.25 * (joints[:, [9, 12]] - joints[:, [12, 9]])
                + 0.5 * (joints[:, [8]] - 0.5 * (joints[:, [9, 12]] + joints[:, [12, 9]]))
            )
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, smpl_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        smpl_output.joints = joints
        return smpl_output
