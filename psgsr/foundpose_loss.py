"""FoundPose-style local RGB-to-normal direction evidence."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


INPUT_SIZE = 420
MAX_MATCHES = 300
POSE_REGULARIZATION_WEIGHT = 0.015


def square_object_crop(
    image, mask, intrinsic, *, size=INPUT_SIZE, padding_fraction=0.2,
    interpolation=cv2.INTER_LINEAR,
):
    """Crop an object to a padded square and update its camera intrinsic."""
    image = np.asarray(image)
    mask = np.asarray(mask) > 0
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("FoundPose direction loss requires a non-empty mask")
    width = int(columns.max() - columns.min() + 1)
    height = int(rows.max() - rows.min() + 1)
    side = max(width, height) * (1.0 + 2.0 * float(padding_fraction))
    center_x = 0.5 * float(columns.min() + columns.max())
    center_y = 0.5 * float(rows.min() + rows.max())
    scale = float(size) / side
    affine = np.asarray([
        [scale, 0.0, -(center_x - 0.5 * side) * scale],
        [0.0, scale, -(center_y - 0.5 * side) * scale],
    ], dtype=np.float32)
    crop = cv2.warpAffine(
        image, affine, (size, size), flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    crop_mask = cv2.warpAffine(
        mask.astype(np.uint8), affine, (size, size),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    transform = np.eye(3, dtype=np.float32)
    transform[:2] = affine
    return (
        crop, crop_mask,
        transform @ np.asarray(intrinsic, dtype=np.float32), affine,
    )


def evidence_loss(mean_similarity, inlier_count, max_matches=MAX_MATCHES):
    """Map correspondence quality and geometric support to a [0, 1] loss."""
    evidence = max(0.0, float(mean_similarity)) * math.log1p(inlier_count)
    confidence = np.clip(evidence / math.log1p(max_matches), 0.0, 1.0)
    return float(1.0 - confidence), float(evidence)


def pose_regularized_loss(loss, regularizer):
    """Attenuate match confidence with an unbounded geometric penalty."""
    regularizer = float(regularizer)
    if not math.isfinite(regularizer):
        return 1.0
    confidence = 1.0 - float(np.clip(loss, 0.0, 1.0))
    return float(1.0 - confidence * math.exp(
        -POSE_REGULARIZATION_WEIGHT * max(0.0, regularizer)
    ))


def _masked_patch_features(feature_map, mask, coordinates=None):
    import torch

    height, width = feature_map.shape[1:]
    xs = (np.arange(width, dtype=np.float32) + 0.5) * mask.shape[1] / width
    ys = (np.arange(height, dtype=np.float32) + 0.5) * mask.shape[0] / height
    grid_x, grid_y = np.meshgrid(xs, ys)
    point_x = np.clip(grid_x.astype(np.int32), 0, mask.shape[1] - 1)
    point_y = np.clip(grid_y.astype(np.int32), 0, mask.shape[0] - 1)
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    valid = eroded[point_y, point_x]
    points = np.column_stack((grid_x[valid], grid_y[valid])).astype(np.float32)
    features = feature_map.permute(1, 2, 0)[
        torch.from_numpy(valid).to(feature_map.device)
    ]
    object_points = None
    if coordinates is not None:
        object_points = np.asarray(coordinates)[
            point_y[valid], point_x[valid]
        ].astype(np.float32)
        keep = (
            np.isfinite(object_points).all(axis=1)
            & (np.linalg.norm(object_points, axis=1) > 1e-8)
        )
        points = points[keep]
        features = features[torch.from_numpy(keep).to(features.device)]
        object_points = object_points[keep]
    return points, features, object_points


def affine_pose_regularizer(query_points, template_points, image_size):
    """Measure how far robust 2D correspondences move from the candidate pose."""
    query = np.asarray(query_points, dtype=np.float32) / float(image_size)
    template = np.asarray(template_points, dtype=np.float32) / float(image_size)
    if len(query) < 3 or len(template) != len(query):
        return None
    cv2.setRNGSeed(0)
    affine, inliers = cv2.estimateAffinePartial2D(
        template, query, method=cv2.RANSAC, ransacReprojThreshold=0.03,
        maxIters=400, confidence=0.99, refineIters=10,
    )
    if affine is None or inliers is None:
        return None
    scale = float(np.hypot(affine[0, 0], affine[1, 0]))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    angle = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
    translation = float(np.linalg.norm(affine[:, 2]))
    regularizer = (
        (angle / 15.0) ** 2
        + (translation / 0.05) ** 2
        + (math.log(scale) / 0.15) ** 2
    )
    return {
        "affine_inlier_count": int(inliers[:, 0].sum()),
        "affine_angle_deg": angle,
        "affine_scale": scale,
        "affine_relative_translation": translation,
        "affine_pose_regularizer": float(regularizer),
    }


def local_pnp_evidence(
    query_points, query_features, template_features,
    template_object_points, intrinsic, *, template_points=None,
    image_size=None, ransac_lock=None,
):
    """Return mutual DINO match quality supported by one PnP pose."""
    import torch

    if not len(query_features) or not len(template_features):
        return {
            "mutual_match_count": 0, "pnp_inlier_count": 0,
            "mean_inlier_similarity": 0.0,
            "evidence_score": 0.0, "loss": 1.0,
            "regularized_loss": 1.0,
        }
    similarity = query_features @ template_features.T
    query_to_template = similarity.argmax(dim=1)
    template_to_query = similarity.argmax(dim=0)
    query_ids = torch.arange(len(query_points), device=similarity.device)
    mutual = template_to_query[query_to_template] == query_ids
    query_ids = query_ids[mutual]
    template_ids = query_to_template[mutual]
    scores = similarity[query_ids, template_ids]
    if len(scores) > MAX_MATCHES:
        keep = scores.topk(MAX_MATCHES).indices
        query_ids, template_ids, scores = (
            query_ids[keep], template_ids[keep], scores[keep]
        )
    image_points = query_points[query_ids.cpu().numpy()]
    object_points = template_object_points[template_ids.cpu().numpy()]
    matched_template_points = (
        np.asarray(template_points)[template_ids.cpu().numpy()]
        if template_points is not None else None
    )
    if len(image_points) < 6:
        return {
            "mutual_match_count": int(len(image_points)),
            "pnp_inlier_count": 0, "mean_inlier_similarity": 0.0,
            "evidence_score": 0.0, "loss": 1.0,
            "regularized_loss": 1.0,
        }

    def solve():
        cv2.setRNGSeed(0)
        return cv2.solvePnPRansac(
            objectPoints=object_points, imagePoints=image_points,
            cameraMatrix=np.asarray(intrinsic, dtype=np.float32),
            distCoeffs=None, iterationsCount=400, reprojectionError=10.0,
            confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if ransac_lock is None:
        success, _, _, inliers = solve()
    else:
        with ransac_lock:
            success, _, _, inliers = solve()
    inlier_count = int(len(inliers)) if success and inliers is not None else 0
    mean_similarity = (
        float(scores[inliers[:, 0]].mean().item()) if inlier_count else 0.0
    )
    loss, evidence = evidence_loss(mean_similarity, inlier_count)
    if matched_template_points is None or image_size is None:
        affine = None
    elif ransac_lock is None:
        affine = affine_pose_regularizer(
            image_points, matched_template_points, image_size,
        )
    else:
        with ransac_lock:
            affine = affine_pose_regularizer(
                image_points, matched_template_points, image_size,
            )
    regularized_loss = (
        pose_regularized_loss(loss, affine["affine_pose_regularizer"])
        if affine is not None else 1.0
    )
    return {
        "mutual_match_count": int(len(image_points)),
        "pnp_inlier_count": inlier_count,
        "pnp_inlier_ratio": float(inlier_count / max(len(image_points), 1)),
        "mean_inlier_similarity": mean_similarity,
        "evidence_score": evidence,
        "loss": loss,
        "regularized_loss": float(regularized_loss),
        **(affine or {
            "affine_inlier_count": 0,
            "affine_angle_deg": None,
            "affine_scale": None,
            "affine_relative_translation": None,
            "affine_pose_regularizer": None,
        }),
    }


class FoundPoseDirectionScorer:
    """Rank rendered-normal candidates with local DINO and PnP evidence."""

    backend = "foundpose_dinov2_local_rgb_normal_pnp"
    loss_formula = (
        "1-mean_cosine(PnP_inliers)*log1p(PnP_inliers)/log1p(300)"
    )
    requires_full_resolution = True
    requires_object_coordinates = True
    pose_regularization_weight = 0.0

    def __init__(
        self, device, input_size=INPUT_SIZE, *, model_size="small",
        dino_root=None,
    ):
        self.device = device
        self.input_size = int(input_size)
        self.model_size = str(model_size)
        self.dino_root = Path(dino_root).resolve() if dino_root else None
        self.feature_layer = 9 if self.model_size == "small" else 18
        if self.input_size <= 0 or self.input_size % 14:
            raise ValueError("FoundPose input size must be a positive multiple of 14")
        if self.model_size not in ("small", "large"):
            raise ValueError("FoundPose model size must be small or large")
        if self.model_size == "large" and self.dino_root is None:
            raise ValueError("FoundPose ViT-L requires the local DINO root")
        self._model = None
        self._load_lock = threading.Lock()
        self._ransac_lock = threading.Lock()
        self._batch_size = 8

    def _load(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            import torch

            if self.model_size == "large":
                checkpoint = self.dino_root / "checkpoints/dinov2_vitl14_pretrain.pth"
                roma_root = self.dino_root / "third_party/RoMa_minima"
                for path in (checkpoint, roma_root):
                    if not path.exists():
                        raise FileNotFoundError(
                            f"FoundPose ViT-L runtime asset is missing: {path}"
                        )
                sys.path.insert(0, str(roma_root))
                from romatch.models.transformer import vit_large

                model = vit_large(
                    img_size=518, patch_size=14, init_values=1.0,
                    ffn_layer="mlp", block_chunks=0,
                ).to(self.device)
                model.load_state_dict(torch.load(checkpoint, map_location=self.device))
                if torch.device(self.device).type == "cuda":
                    model = model.half()
            else:
                local_root = os.environ.get("DINOV2_ROOT")
                if not local_root:
                    cached_root = (
                        Path(torch.hub.get_dir())
                        / "facebookresearch_dinov2_main"
                    )
                    if cached_root.is_dir():
                        local_root = str(cached_root)
                if local_root:
                    model = torch.hub.load(
                        local_root, "dinov2_vits14_reg", pretrained=True,
                        source="local",
                    )
                else:
                    model = torch.hub.load(
                        "facebookresearch/dinov2", "dinov2_vits14_reg",
                        pretrained=True, trust_repo=True,
                    )
            self._model = model.to(self.device).eval()
        return self._model

    def _feature_batch(self, images):
        import torch
        import torch.nn.functional as functional

        model = self._load()
        tensor = torch.from_numpy(np.ascontiguousarray(np.stack(images))).permute(
            0, 3, 1, 2,
        ).to(device=self.device, dtype=torch.float32) / 255.0
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device,
        )[None, :, None, None]
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device,
        )[None, :, None, None]
        normalized = ((tensor - mean) / std).to(
            dtype=next(model.parameters()).dtype,
        )
        with torch.inference_mode():
            features = model.prepare_tokens_with_masks(normalized)
            for index, block in enumerate(model.blocks):
                features = block(features)
                if index == self.feature_layer:
                    break
            features = model.norm(features)
            features = features[:, 1 + int(getattr(model, "num_register_tokens", 0)):]
        grid_size = self.input_size // 14
        features = features.reshape(
            len(images), grid_size, grid_size, -1,
        ).permute(0, 3, 1, 2)
        return list(functional.normalize(features, dim=1).unbind(0))

    def _feature_maps(self, images):
        import torch

        batch_size = min(self._batch_size, len(images))
        while True:
            try:
                output = []
                for start in range(0, len(images), batch_size):
                    output.extend(self._feature_batch(
                        images[start:start + batch_size],
                    ))
                self._batch_size = batch_size
                return output
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                self._batch_size = batch_size
                torch.cuda.empty_cache()

    def score_candidates(
        self, reference_key, reference_rgb, reference_mask, camera, candidates,
    ):
        if not candidates:
            return {}
        padding = 0.1 if reference_key == "top" else 0.2
        reference, reference_crop_mask, intrinsic, _ = square_object_crop(
            reference_rgb, reference_mask, camera["intrinsic"],
            size=self.input_size, padding_fraction=padding,
        )
        prepared = []
        for key, candidate in candidates.items():
            if "object_coordinates" not in candidate:
                raise ValueError("FoundPose candidate is missing object coordinates")
            normal = np.asarray(candidate["normal_map"], dtype=np.uint8).copy()
            normal[~(np.asarray(candidate["mask"]) > 0)] = 0
            crop, crop_mask, _, affine = square_object_crop(
                normal, candidate["mask"], camera["intrinsic"],
                size=self.input_size, padding_fraction=padding,
            )
            coordinates = cv2.warpAffine(
                np.asarray(candidate["object_coordinates"], dtype=np.float32),
                affine, (self.input_size, self.input_size),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0.0, 0.0, 0.0),
            )
            prepared.append((key, crop, crop_mask, coordinates))
        start = time.perf_counter()
        feature_maps = self._feature_maps([
            reference, *[item[1] for item in prepared],
        ])
        inference_seconds = time.perf_counter() - start
        query_points, query_features, _ = _masked_patch_features(
            feature_maps[0], reference_crop_mask,
        )
        results = {}
        for feature_map, (key, _, mask, coordinates) in zip(
            feature_maps[1:], prepared,
        ):
            _, template_features, object_points = _masked_patch_features(
                feature_map, mask, coordinates,
            )
            result = local_pnp_evidence(
                query_points, query_features, template_features,
                object_points, intrinsic, ransac_lock=self._ransac_lock,
            )
            results[key] = {
                **result,
                "foreground_mean_confidence": float(1.0 - result["loss"]),
                "backend": self.backend,
                "reference_modality": "real_rgb",
                "candidate_modality": "rendered_camera_normal",
                "selection_source": "foundpose_local_rgb_normal_pnp_evidence",
                "input_size": self.input_size,
                "padding_fraction": padding,
                "feature_layer": self.feature_layer,
                "model_size": self.model_size,
                "inference_seconds": float(inference_seconds),
                "inference_batch_size": self._batch_size,
            }
        return results
