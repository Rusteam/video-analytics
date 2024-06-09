"""Use fiftyone as the main data visualization tool."""

from pathlib import Path

import fiftyone as fo
import fiftyone.types as fot
import fiftyone.utils.ultralytics as fouu
import pydantic
from termcolor import cprint
from ultralytics import YOLO


class UltralyticsModel(pydantic.BaseModel):
    name: str = pydantic.Field("yolov8s.pt", description="Ultralytics model name")
    # classes: list[str] = pydantic.Field(
    #     ["person"], description="List of classes to detect"
    # )
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


if __name__ == "__main__":
    FiftyoneDataset(name="AIQ").track(label_field="yolob8s")
