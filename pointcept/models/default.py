import torch
import torch.nn as nn

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        conditions=None,
        teacher=None,
        teacher_weights=None,
        teacher_criteria=None,
    ):
        super().__init__()
        self.conditions = conditions if conditions is not None else ("Default",)
        self.num_classes = (
            num_classes if isinstance(num_classes, (list, tuple)) else (num_classes,)
        )
        assert len(self.conditions) == len(self.num_classes)
        self.seg_heads = nn.ModuleDict(
            {
                condition: (
                    nn.Linear(backbone_out_channels, self.num_classes[i])
                    if self.num_classes[i] > 0
                    else nn.Identity()
                )
                for i, condition in enumerate(self.conditions)
            }
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.teacher = (
            DefaultTeacherWrapper(teacher, teacher_weights)
            if teacher is not None
            else None
        )
        self.teacher_criteria = (
            build_criteria(teacher_criteria)
            if teacher_criteria is not None
            else self.criteria
        )
        if self.teacher is not None:
            self.distill_heads = nn.ModuleDict(
                {
                    condition: (
                        nn.Linear(backbone_out_channels, self.num_classes[i])
                        if self.num_classes[i] > 0
                        else nn.Identity()
                    )
                    for i, condition in enumerate(self.conditions)
                }
            )

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point

        if not self.training:
            condition = input_dict.get("condition", ("Default",))
            # For evaluation, we assume homogenous batches.
            assert all(cond == condition[0] for cond in condition)
            assert condition[0] in self.conditions
            seg_head = self.seg_heads[condition[0]]
            seg_logits = seg_head(feat)
            # eval
            if "segment" in input_dict.keys():
                loss = self.criteria(seg_logits, input_dict["segment"])
                return_dict["loss"] = loss
                return_dict["seg_logits"] = seg_logits
            # test
            else:
                return_dict["seg_logits"] = seg_logits
        # train
        else:
            if self.teacher is not None:
                teacher_feat = self.teacher._forward_backbone(input_dict)

            condition_mask = dict()
            if "condition" not in input_dict.keys():
                condition_mask["Default"] = torch.ones(
                    feat.shape[0], dtype=torch.bool, device=feat.device
                )
            else:
                unmixed_offset = input_dict["unmixed_offset"].detach().cpu().tolist()
                assert len(input_dict["condition"]) == len(unmixed_offset)
                start = 0
                for condition, end in zip(input_dict["condition"], unmixed_offset):
                    assert condition in self.conditions

                    if condition not in condition_mask:
                        condition_mask[condition] = torch.zeros(
                            feat.shape[0], dtype=torch.bool, device=feat.device
                        )

                    condition_mask[condition][start:end] = True
                    start = end

            loss_list = []
            for condition, mask in condition_mask.items():
                logits = self.seg_heads[condition](feat[mask])
                loss = self.criteria(logits, input_dict["segment"][mask])
                if not torch.isfinite(loss).all():
                    loss = logits.sum() * 0.0

                if self.teacher is not None:
                    distill_logits = self.distill_heads[condition](feat[mask])
                    teacher_labels = self.teacher._forward_seg_head(
                        teacher_feat[mask], condition
                    )
                    distill_loss = self.teacher_criteria(distill_logits, teacher_labels)
                    loss = 0.5 * loss + 0.5 * distill_loss
                loss_list.append(loss * mask.sum())
            return_dict["loss"] = torch.stack(loss_list).sum() / feat.shape[0]
        return return_dict


class DefaultTeacherWrapper:  # not an nn.Module, keeps it detached from training
    def __init__(
        self,
        teacher,
        teacher_weights,
    ):
        super().__init__()
        self.teacher = build_model(teacher).cuda()

        # Load weights
        ckpt = torch.load(teacher_weights, map_location="cpu", weights_only=False)
        state_dict = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in ckpt["state_dict"].items()
        }
        self.teacher.load_state_dict(state_dict, strict=True)

        self.teacher.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def _forward_backbone(self, input_dict):
        point = Point(input_dict)
        point = self.teacher.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        return feat

    @torch.no_grad()
    def _forward_seg_head(self, feat, condition):
        if hasattr(self.teacher, "seg_heads"):
            seg_head = self.teacher.seg_heads[condition]
            teacher_logits = seg_head(feat)
        else:
            teacher_logits = feat
        return teacher_logits.argmax(dim=-1)
