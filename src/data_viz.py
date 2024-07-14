"""Use fiftyone as the main data visualization tool."""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import fiftyone as fo
import fiftyone.types as fot
import fiftyone.utils.ultralytics as fouu
import fiftyone.zoo as foz
import numpy as np
import pydantic
import yaml
from termcolor import cprint
from tqdm import tqdm
from ultralytics import YOLO


def _load_yaml(path: str):
    with open(path) as fp:
        return yaml.safe_load(fp)


def _boxes_match(src_detection: list[float], target_detection: list[float]):
    """Check if two bboxes overlap.

    Two bboxes overlap if the center of the source
    bbox is within the bounds of the target_bbox.
    """
    x, y, w, h = src_detection.bounding_box
    zone_x, zone_y, zone_w, zone_h = target_detection.bounding_box
    bbox_center = (x + w / 2, y + h / 2)
    within_x = zone_x < bbox_center[0] < zone_x + zone_w
    within_y = zone_y < bbox_center[1] < zone_y + zone_h
    return int(within_x and within_y)


def _to_boxmot_detection(
    detection: fo.Detection, width: int, height: int, class_labels: list[str]
) -> list[float]:
    """Convert fiftyone detection to boxmot format.

    Expected format: (x, y, x, y, conf, cls)
    """
    x0, y0, w, h = detection.bounding_box
    x_min, y_min = x0 * width, y0 * height
    x_max, y_max = (x0 + w) * width, (y0 + h) * height

    return [
        x_min,
        y_min,
        x_max,
        y_max,
        detection.confidence,
        class_labels.index(detection.label),
    ]


def _from_boxmot_tracklet(
    tracklet: np.ndarray, width: int, height: int, class_labels: list[str]
) -> fo.Detection:
    """Convert boxmot tracklet to fiftone detection format.

    Input format: (x, y, x, y, id, conf, cls, index)
    """
    x_min, y_min, x_max, y_max, idx, conf, class_id, track_id = tracklet

    x0, y0 = x_min / width, y_min / height
    w, h = (x_max - x_min) / width, (y_max - y_min) / height

    return fo.Detection(
        bounding_box=[x0, y0, w, h],
        confidence=conf,
        label=class_labels[int(class_id)],
        index=int(idx),
    )


class UltralyticsModel(pydantic.BaseModel):
    model: str = pydantic.Field("yolov8s.pt", description="Ultralytics model name")
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640

    @property
    def detector(self):
        return YOLO(self.model, task="detect")

    @property
    def pose_estimator(self):
        if "pose" not in self.model:
            size, ext = self.model.split(".")
            self.model = f"{size}-pose.{ext}"
        return YOLO(self.model)

    @property
    def keypoint_skeleton(self):
        return fo.KeypointSkeleton(
            labels=[
                "nose",
                "left eye",
                "right eye",
                "left ear",
                "right ear",
                "left shoulder",
                "right shoulder",
                "left elbow",
                "right elbow",
                "left wrist",
                "right wrist",
                "left hip",
                "right hip",
                "left knee",
                "right knee",
                "left ankle",
                "right ankle",
            ],
            edges=[
                [11, 5, 3, 1, 0, 2, 4, 6, 12],
                [9, 7, 5, 6, 8, 10],
                [15, 13, 11, 12, 14, 16],
            ],
        )

    def track(self, path: str | Path):
        return self.detector.track(
            path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            stream=True,
            show=False,
        )

    def detect_keypoints(self, path: str | Path):
        return self.pose_estimator(path, stream=True, show=False)


class BoxmotTracker(pydantic.BaseModel):
    """Boxmot tracking and reid configuration.

    More details @ https://github.com/mikel-brostrom/boxmot
    """

    tracking_method: str = "botsort"
    reid_model: str = "osnet_x0_25_market1501.pt"
    fp16: bool = False
    max_age: int = 300

    @property
    def device(self):
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @property
    def tracker(self):
        from boxmot import BoTSORT, DeepOCSORT, StrongSORT

        trackers = dict(botsort=BoTSORT, deepocsort=DeepOCSORT, strongsort=StrongSORT)
        if self.tracking_method not in trackers:
            raise ValueError(
                f"Wrong tracking method passed, available {trackers.keys()}"
            )
        return trackers[self.tracking_method](
            model_weights=Path(self.reid_model),
            device=self.device,
            fp16=self.fp16,
            max_age=self.max_age,
        )


class AnnotatedZone(pydantic.BaseModel):
    smp_id: str
    cashier: list[float]
    exit: list[float]

    @pydantic.field_validator("cashier", "exit")
    def check_values(val: list[float]):
        if not len(val) == 4:
            raise ValueError("wrong number of inputs in the bbox")
        if not all([0 <= i <= 1 for i in val]):
            raise ValueError("values must be between 0 and 1")
        return val

    def to_detections(self):
        """convert zone to fiftyone detection format"""
        dets = []
        for zone in ["cashier", "exit"]:
            x0, y0, x1, y1 = getattr(self, zone)
            dets.append(
                fo.Detection(label=zone, bounding_box=[x0, y0, x1 - x0, y1 - y0])
            )
        return fo.Detections(detections=dets)


class FiftyoneDataset(pydantic.BaseModel):
    name: str = pydantic.Field(..., description="name of fiftyone dataset")
    overwrite: bool = pydantic.Field(
        False, description="whether to overwrite the dataset if already exists"
    )
    # created in the code
    dataset: Optional[fo.Dataset] = None

    class Config:
        arbitrary_types_allowed = True

    def create(self, path: str):
        """Create a fiftyone dataset from a folder of video files"""
        path = Path(path)
        if not path.exists() or not path.is_dir():
            raise NotADirectoryError("directory does not exist or not a directory")

        if fo.dataset_exists(self.name) and self.overwrite:
            fo.delete_dataset(self.name, verbose=True)

        self.dataset = fo.Dataset.from_dir(
            dataset_dir=path.as_posix(),
            dataset_type=fot.VideoDirectory,
            persistent=True,
            name=self.name,
        )
        cprint(f"Successfully created a dataset with {len(self.dataset)}")

    def track(self, label_field: str = "detections", debug: bool = False, **kwargs):
        """Apply yolov8 detector and tracker"""
        model = UltralyticsModel(**kwargs)
        self._load_dataset()
        for smp in self.dataset.select_fields():
            results = model.track(smp.filepath)
            frame_no = 1
            for res in results:
                detections = fouu.to_detections(res)
                smp.frames[frame_no][label_field] = detections
                frame_no += 1
                if debug and frame_no > 20:
                    break
            smp.save()
        cprint(f"Finished tracking for {len(self.dataset)} samples", "green")

    def track_reid(
        self, label_field: str, new_field: str, debug: bool = False, **kwargs
    ):
        """Apply boxmot tracker with a reid model on top of existing detections."""
        tracker = BoxmotTracker(**kwargs).tracker
        self._load_dataset()
        self.dataset.compute_metadata()
        for smp in self.dataset.select_fields(f"frames.{ label_field }"):
            vidoe_capture = self._get_video_capture(smp)
            frame_no = 1
            for frame_id in tqdm(smp.frames, desc="Iterating frames"):
                detections = self._as_boxmot_detections(smp, frame_id, label_field)
                img = self._read_video_frame(vidoe_capture, frame_id - 1)
                tracks = tracker.update(np.array(detections), img)
                smp.frames[frame_id][new_field] = self._from_boxmot_tracklets(
                    tracks, smp, label_field
                )

                # debug
                frame_no += 1
                if debug and frame_no > 100:
                    break
            smp.save()
        cprint(f"Finished tracking for {len(self.dataset)} samples", "green")

    def detect_keypoints(self, label_field: str = "keypoints", **kwargs):
        """Apply yolov8 detector and tracker"""
        model = UltralyticsModel(**kwargs)
        self._load_dataset()
        self.dataset.default_skeleton = model.keypoint_skeleton
        for smp in self.dataset.select_fields():
            results = model.track(smp.filepath)
            frame_no = 1
            for res in results:
                keypoints = fouu.to_keypoints(res, confidence_thresh=0.4)
                for det, idx in zip(keypoints.keypoints, res.boxes.id):
                    det.index = idx
                smp.frames[frame_no][label_field] = keypoints
                frame_no += 1
            smp.save()
        cprint(f"Finished pose estimation for {len(self.dataset)} samples", "green")

    def annotate_zones(self, annotations: str, label_field: str = "zone"):
        """Use config file with zone annotations to add to fiftyone"""
        self._load_dataset()
        self._delete_field_if_exists(label_field)

        zone_config = _load_yaml(annotations).get("zones", [])
        for conf in zone_config:
            zone = AnnotatedZone(**conf)
            smp = self.dataset[zone.smp_id]
            for frame_id in smp.frames:
                smp.frames[frame_id][label_field] = zone.to_detections()
            smp.save()
        cprint(f"Annotated zones for {len(zone_config)} samples", "green")

    def identify_customer(
        self,
        tracking_field: str,
        zone_field: str,
        new_field: str = "visitor_type",
        iou_threshold: float = 0.5,
    ):
        """Classify each track id as a customer or cashier base on zone intersection"""
        self._load_dataset()
        self._delete_field_if_exists(new_field)
        view = self._filter_labels(tracking_field, "person")

        # count iou matching with cashier
        n_matches = {}
        for smp in view.select_fields(
            [f"frames.{ tracking_field }", f"frames.{ zone_field }"]
        ):
            n_matches[smp.id] = defaultdict(lambda: 0)
            for frame_id in smp.frames:
                detections = smp[frame_id][tracking_field].detections
                zone = self._get_zone_bbox(smp[frame_id], zone_field, "cashier")
                for det in detections:
                    n_matches[smp.id][det.index] += _boxes_match(det, zone)

        # update class labels for customer and cashiers
        for smp in view.select_fields(f"frames.{tracking_field}"):
            smp_view = self.dataset.select(smp.id)
            totals = self._count_frame_values(smp_view, tracking_field, "index")
            for frame_id in smp.frames:
                detections = []
                for det in smp.frames[frame_id][tracking_field].detections:
                    match_ratio = (
                        n_matches[smp.id][det.index] / totals[det.index] > iou_threshold
                    )
                    label = "cashier" if match_ratio else "customer"
                    detections.append(
                        fo.Detection(
                            label=label,
                            bounding_box=det.bounding_box,
                            confidence=det.confidence,
                            index=det.index,
                        )
                    )
                smp[frame_id][new_field] = fo.Detections(detections=detections)
            smp.save()

            # print counts of customers
            for label in ["cashier", "customer"]:
                distinct_vals = self._get_distinct_indices(smp_view, new_field, label)
                cprint(f"{smp.id} {label} count: {len(distinct_vals)}", "green")

    def identify_exit(self, tracking_field: str, zone_field: str):
        """Identify customers exiting the shop.

        Exit is defined as an intersection with the exit zone
        for last frame for a customer track id.
        """
        self._load_dataset()

        for smp in self.dataset:
            smp_view = self.dataset.select(smp.id)
            customer_tracks = self._get_distinct_indices(
                smp_view, tracking_field, "customer"
            )

            # TODO add number of customers entering
            n_exit = 0
            cur_frame = smp.metadata["total_frame_count"]
            while len(customer_tracks) > 0 and cur_frame > 1:
                if not smp[cur_frame][tracking_field]:
                    cur_frame -= 1
                    continue

                # check if last customer frame is in the exit box
                for det in smp[cur_frame][tracking_field].detections:
                    if det.index in customer_tracks:
                        # FIXME optimize time complexity
                        customer_tracks.remove(det.index)
                        zone = self._get_zone_bbox(smp[cur_frame], zone_field, "exit")
                        n_exit += _boxes_match(det, zone)
                cur_frame -= 1

            cprint(f"{smp.id}: Number of customers exiting is {n_exit}", "green")

    def classify_customers(
        self,
        patch_field: str = "visitor_type",
        export_dir: str = "./tmp/patches",
        overwrite: bool = False,
    ):
        """Classify each patch with a clip model"""
        patches_dataset = self._create_patches(
            patch_field, "customer", export_dir, overwrite
        )
        # classify patches
        clip = foz.load_zoo_model(
            "clip-vit-base32-torch", classes=["adult man", "adult woman", "girl", "boy"]
        )
        patches_dataset.apply_model(clip, label_field="clip", batch_size=4)
        cprint(f"Finished predicting on {len(patches_dataset)} samples", "yellow")

        # print top value for each customer
        customer_ids = patches_dataset.distinct("customer_id.label")
        for idx in customer_ids:
            label_counts = patches_dataset.match(
                fo.ViewField("customer_id.label") == idx
            ).count_values("clip.label")
            num_predictions = sum(label_counts.values())
            class_probs = {
                k: round(v / num_predictions, 2) for k, v in label_counts.items()
            }
            cprint(f"Customer {idx!r} class probabilities: {class_probs}", "green")

    def cashier_hand(
        self,
        patch_field: str = "visitor_type",
        export_dir: str = "./tmp/patches",
        overwrite: bool = False,
        **kwargs,
    ):
        """Detect if a cashier is right-handed or left-handed."""
        patches_dataset = self._create_patches(
            patch_field, "cashier", export_dir, overwrite
        ).take(
            200
        )  # TODO find a way to select best samples

        # classify patches
        model = UltralyticsModel(**kwargs)
        patches_dataset.apply_model(
            model.pose_estimator,
            label_field="keypoints",
            batch_size=4,
            confidence_thresh=kwargs.get("conf_threshold", 0.5),
        )
        patches_dataset.default_skeleton = model.keypoint_skeleton
        cprint(f"Finished predicting on {len(patches_dataset)} samples", "yellow")

        # retrieve detected keypoints per cashier and compute variances
        right_index = patches_dataset.default_skeleton.labels.index("right wrist")
        left_index = patches_dataset.default_skeleton.labels.index("left wrist")
        customer_ids = patches_dataset.distinct("cashier_id.label")
        for idx in customer_ids:
            view = patches_dataset.match(fo.ViewField("cashier_id.label") == idx)
            points = view.values("keypoints.keypoints.points")
            # TODO consider confidence threshold and exclude zeros
            # select x,y for each hand and compute variance across each dimension
            left_points = np.array(
                [p[0][left_index] for p in points if p and sum(p[0][left_index]) > 0]
            )
            right_points = np.array(
                [p[0][right_index] for p in points if p and sum(p[0][right_index]) > 0]
            )
            left_var = (np.var(left_points[0]) + np.var(left_points[1])) / 2
            right_var = (np.var(right_points[0]) + np.var(right_points[1])) / 2

            cashier_hand = "left" if left_var > right_var else "right"
            cprint(
                f"Cashier {idx!r} is {cashier_hand!r}-handed: {left_var=:.2f}, {right_var=:.2f}",
                "green",
            )

    def _load_dataset(self):
        if self.name in fo.list_datasets():
            self.dataset = fo.load_dataset(self.name)
        else:
            raise KeyError("Dataset {self.name=!r} does not exist")

    def _create_patches(
        self, patch_field: str, label: str, export_dir: str, overwrite: bool
    ):
        """Export detetion patches to disk and create a new dataset"""

        patches_dataset_name = f"{self.name}-{label}-patches"
        if fo.dataset_exists(patches_dataset_name) and overwrite:
            fo.delete_dataset(patches_dataset_name, verbose=True)

        if fo.dataset_exists(patches_dataset_name):
            patches_dataset = fo.load_dataset(patches_dataset_name)
        else:
            # export patches
            self._load_dataset()
            customer_view = self._filter_labels(patch_field, label=label)
            patches = customer_view.to_frames(sample_frames=True).to_patches(
                patch_field
            )
            for p in patches:
                p[patch_field].label = f"{p[patch_field].label}-{p[patch_field].index}"
                p.save()
            patches.export(
                export_dir=export_dir,
                dataset_type=fot.ImageClassificationDirectoryTree,
                label_field=patch_field,
            )
            patches_dataset = fo.Dataset.from_dir(
                export_dir,
                dataset_type=fot.ImageClassificationDirectoryTree,
                label_field=f"{label}_id",
                name=patches_dataset_name,
                persistent=True,
            )
        if len(patches_dataset) == 0:
            raise ValueError("no samples in the dataset")
        return patches_dataset

    def _delete_field_if_exists(self, label_field: str):
        if not self.dataset:
            raise RuntimeError("make sure to load the dataset first")

        if self.dataset.has_frame_field(label_field):
            self.dataset.delete_frame_field(label_field)

    def _filter_labels(
        self,
        label_field: str,
        label: str,
        key: str = "label",
        view: None | fo.DatasetView = None,
    ):
        view = view or self.dataset
        return view.filter_labels(f"frames.{ label_field }", fo.ViewField(key) == label)

    @staticmethod
    def _get_zone_bbox(smp_frame: fo.Sample, zone_field: str, label: str):
        return list(
            filter(
                lambda x: x.label == label,
                smp_frame[zone_field].detections,
            )
        )[0]

    @staticmethod
    def _count_frame_values(
        view: fo.DatasetView,
        label_field: str,
        value: str,
        label_type: str = "detections",
    ):
        return view.count_values(f"frames.{label_field}.{label_type}.{value}")

    @staticmethod
    def _get_distinct_indices(view: fo.DatasetView, label_field: str, label: str):
        return view.filter_labels(
            f"frames.{label_field}", fo.ViewField("label") == label
        ).distinct(f"frames.{label_field}.detections.index")

    @staticmethod
    def _get_video_capture(smp: fo.Sample):
        cap = cv2.VideoCapture(smp.filepath)
        if not cap.isOpened():
            raise ValueError(f"Unable to read a video file at {smp.filepath!r}")
        else:
            return cap

    @staticmethod
    def _read_video_frame(
        cap: cv2.VideoCapture, frame_number: int
    ) -> np.ndarray | None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, img = cap.read()
        if ret:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print(f"Could not read frame {frame_number}")

    def _as_boxmot_detections(
        self, smp: fo.Sample, frame_id: int, label_field: str
    ) -> list[list[float]]:
        detections = smp[frame_id][label_field].detections
        return [
            _to_boxmot_detection(
                det,
                width=smp.metadata["frame_width"],
                height=smp.metadata["frame_height"],
                class_labels=self._get_class_labels(f"frames.{ label_field }"),
            )
            for det in detections
        ]

    def _from_boxmot_tracklets(
        self, tracks: np.ndarray, smp: fo.Sample, label_field: str
    ) -> fo.Detections:
        return fo.Detections(
            detections=[
                _from_boxmot_tracklet(
                    tracklet,
                    width=smp.metadata["frame_width"],
                    height=smp.metadata["frame_height"],
                    class_labels=self._get_class_labels(f"frames.{ label_field }"),
                )
                for tracklet in tracks
            ]
        )

    def _get_class_labels(self, label_field: str) -> list[str]:
        return self.dataset.distinct(f"{label_field}.detections.label")
