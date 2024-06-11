"""Use fiftyone as the main data visualization tool."""

from collections import defaultdict
from pathlib import Path

import fiftyone as fo
import fiftyone.types as fot
import fiftyone.utils.ultralytics as fouu
import fiftyone.zoo as foz
import pydantic
import yaml
from termcolor import cprint
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
    dataset: None = None

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

    def _load_dataset(self):
        if self.name in fo.list_datasets():
            self.dataset = fo.load_dataset(self.name)
        else:
            raise KeyError("Dataset {self.name=!r} does not exist")

    def track(self, label_field: str = "detections", **kwargs):
        """Apply yolov8 detector and tracker"""
        model = UltralyticsModel(**kwargs)
        self._load_dataset()
        for smp in self.dataset.select_fields():
            results = model.track(smp.filepath)
            frame_no = 1
            for res in results:
                detections = fouu.to_detections(res)
                for det, idx in zip(detections.detections, res.boxes.id):
                    det.index = idx
                smp.frames[frame_no][label_field] = detections
                frame_no += 1
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
                keypoints = fouu.to_keypoints(res)
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
        patch_field: str,
        export_dir: str = "./data/export/patches",
        overwrite: bool = False,
    ):
        patches_dataset_name = f"{self.name}-patches"
        if fo.dataset_exists(patches_dataset_name) and overwrite:
            fo.delete_dataset(patches_dataset_name, verbose=True)
        elif fo.dataset_exists(patches_dataset_name):
            patches_dataset = fo.load_dataset(patches_dataset_name)
        else:
            # # export patches
            self._load_dataset()
            customer_view = self._filter_labels(patch_field, label="customer")
            patches = customer_view.to_frames(sample_frames=True).to_patches(
                patch_field
            )
            for p in patches:
                p[
                    patch_field
                ].label = f"{ p[patch_field].label }-{p[patch_field].index}"
                p.save()
            patches.export(
                export_dir=export_dir,
                dataset_type=fot.ImageClassificationDirectoryTree,
                label_field=patch_field,
            )
            patches_dataset = fo.Dataset.from_dir(
                export_dir,
                dataset_type=fot.ImageClassificationDirectoryTree,
                label_field="customer_id",
                name=f"{self.name}-patches",
                persistent=True,
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
