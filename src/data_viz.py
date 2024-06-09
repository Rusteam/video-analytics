"""Use fiftyone as the main data visualization tool."""

from collections import defaultdict
from pathlib import Path

import fiftyone as fo
import fiftyone.types as fot
import fiftyone.utils.ultralytics as fouu
import pydantic
import yaml
from termcolor import cprint
from ultralytics import YOLO


def _load_yaml(path: str):
    with open(path) as fp:
        return yaml.safe_load(fp)


class UltralyticsModel(pydantic.BaseModel):
    name: str = pydantic.Field("yolov8s.pt", description="Ultralytics model name")
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640

    @property
    def detector(self):
        return YOLO(
            self.name,
            task="detect",
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

    def annotate_zones(self, annotations: str, label_field: str = "zone"):
        """Use config file with zone annotations to add to fiftyone"""
        self._load_dataset()
        if self.dataset.has_frame_field(label_field):
            self.dataset.delete_frame_field(label_field)

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
        n_matches = {}
        view = self.dataset.filter_labels(
            f"frames.{ tracking_field }", fo.ViewField("label") == "person"
        )
        # count iou matching with cashier
        for smp in view.select_fields(
            [f"frames.{ tracking_field }", f"frames.{ zone_field }"]
        ):
            n_matches[smp.id] = defaultdict(lambda: 0)
            for frame_id in smp.frames:
                detections = smp[frame_id][tracking_field].detections
                zone = list(
                    filter(
                        lambda x: x.label == "cashier",
                        smp[frame_id][zone_field].detections,
                    )
                )[0]
                zone_x, zone_y, zone_w, zone_h = zone.bounding_box
                for det in detections:
                    x, y, w, h = det.bounding_box
                    bbox_center = (x + w / 2, y + h / 2)
                    within_x = zone_x < bbox_center[0] < zone_x + zone_w
                    within_y = zone_y < bbox_center[1] < zone_y + zone_h
                    n_matches[smp.id][det.index] += int(within_x and within_y)

        # update class labels for customer and cashiers
        for smp in view.select_fields(f"frames.{tracking_field}"):
            smp_view = self.dataset.select(smp.id)
            totals = smp_view.count_values(f"frames.{tracking_field}.detections.index")
            for frame_id in smp.frames:
                detections = []
                for det in smp.frames[frame_id][tracking_field].detections:
                    if det.index not in n_matches[smp.id]:
                        continue

                    is_cashier = (
                        n_matches[smp.id][det.index] / totals[det.index] > iou_threshold
                    )
                    label = "cashier" if is_cashier else "customer"
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
                n_distinct = smp_view.filter_labels(
                    f"frames.{new_field}", fo.ViewField("label") == label
                ).distinct(f"frames.{new_field}.detections.index")
                cprint(f"{smp.id} {label} counts: {len(n_distinct)}", "green")
