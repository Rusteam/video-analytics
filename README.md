# Real-time video analytics

The goal of this repo is to build a real-time
video analytics system that is capable of solving
multiple various tasks.

## Features:

- \[x\] Detect all people within a frame
- \[x\] Identify number of unique customers in the shop
- \[x\] Track each customer
- \[x\] Identify how many people entered and exited shop during the video
- \[x\] Identify gender (male/female) and age (child/adult as classification) for each customer
- \[ \] Identify number of unique customers that made a purchase
- \[ \] Identify if cashier is right or left-handed

![example output](./example.gif)

## Usage

### Installation

1. Install [ poetry ](https://python-poetry.org/docs/basic-usage/) and use system env (in case using conda or other):

```
poetry env use system
```

2. Install torch and torchvision with conda (or other):

```
conda install pytorch torchvision -c pytorch
```

3. Install dependencies (poetry is recommended but requirements.txt is also available):

```
poetry install
```

4. Install ffmpeg as per system.

### Tools and packages

- [Fiftyone](https://docs.voxel51.com/index.html) is used as the main tool for managing data
  and labels.
- [Fire](https://google.github.io/python-fire/guide/) is the main cli entrypoint
  to run all scripts.
- User inputs are validated with
  [pydantic](https://docs.pydantic.dev/latest/concepts/models/).

The `FiftyoneDataset` class is the main entrypoint to run all commands from cli.
Check out its usage below:

```
NAME
    main.py FiftyoneDataset - Usage docs: https://docs.pydantic.dev/2.7/concepts/models/

SYNOPSIS
    main.py FiftyoneDataset GROUP | COMMAND | <flags>

FLAGS
    -n, --name=NAME (required)
        Type: str
    -o, --overwrite=OVERWRITE
        Type: bool
        Default: False

```

### Run the commands

#### Create a dataset and track objects

1. Create a fiftyone dataset to visualize data and labels:

```
python main.py FiftyoneDataset --name <name> create <path/to/video/folder>

# keep fiftyone running to observe changes
fiftyone app launch <name>
```

2. Run detection and tracking:

```
python main.py FiftyoneDataset --name <name> track --model yolov8s --label-field yolov8s
```

This command will download a respective yolov8 model to the current directory
and create a label field "yolov8s" that will contain
detections with COCO classes and track ids.

- [Ultralytics yolov8](https://docs.ultralytics.com/tasks/detect/)
  family of models is used for object detection.
- [BoxSort](https://github.com/NirAharon/BoT-SORT) is used to
  connect frame-level detections into tracklets.

#### Count customers

To be able to identify a customer we need to annotate
a cashier's zone in a yaml config file. Once a fiftyone
dataset is created, find out video sample id from web UI
and add an entry to the [ config file ](configs/annotations.yaml)
following the existing example.

1. Add these annotations to fiftyone:

```
python main.py FiftyoneDataset --name <name> annotate_zones configs/annotations.yaml --label_field zone
```

Verify the correctness of zone annotations in the fiftyone app.

2. After adding zones, run the following command to label
   each person as a cashier or a customer:

```
❯ python main.py FiftyoneDataset --name <name> identify_customer --tracking_field yolov8s --zone_field zone --iou-threshold 0.5

Output:
6665610ec7102da65e0f95ef cashier count: 2
6665610ec7102da65e0f95ef customer count: 41
```

A new label field is created that maps a "person" class to either "cashier" or
"customer" based on the intersection with the "cashier" zone.

#### Number of customers exiting the shop

This step assumes that the exit zone has been annotated at
the previous step, as well as customer identification.

```
❯ python main.py FiftyoneDataset --name <name> identify_exit --tracking_field visitor_type --zone_field zone

Output:
6665610ec7102da65e0f95ef: Number of customers exiting is 4
```

An "exit" is defined as a last frame for each tracklet to
intersect with the annotated "exit" zone.

#### Age and Gender prediction

Age and gender classification is handled by the `clip` model with 4 classes:

- adult woman
- adult man
- girl
- boy

The script below will create a new dataset with customer patches,
download the `clip` model using `fiftyone zoo`
and classify each patch into the defined categories. Then,
the predictions will be grouped for each customer and simple voting
will decides the final class probabilities.

```
❯ python main.py FiftyoneDataset --name <name> classify_customers --patch-field visitor_type --export-dir data/interim/patches

Output:
Customer 'customer-1' class probabilities: {'adult man': 0.64, 'boy': 0.18, 'adult w
oman': 0.18}
Customer 'customer-10' class probabilities: {'adult woman': 0.87, 'girl': 0.13}
Customer 'customer-11' class probabilities: {'adult man': 0.62, 'adult woman': 0.38}
Customer 'customer-14' class probabilities: {'adult man': 0.85, 'adult woman': 0.03,
 'boy': 0.13}
Customer 'customer-15' class probabilities: {'adult man': 0.33, 'adult woman': 0.51,
 'girl': 0.16}
Customer 'customer-17' class probabilities: {'adult man': 0.87, 'adult woman': 0.1,
```

It's also possible to review the clip predictions in the fiftyone app
by opening the `<name>-patches` dataset. Filter customers by the `customer_id` field
and review the label counts in the `clip` field.

**Possible improvements:**

1. Select top-k patches per customer to optimize speed
1. Run a face detection algorithm and apply a model trained on
   the [adience dataset](https://paperswithcode.com/dataset/adience)

#### Customer purchase detection (TODO)

There are few steps to tackle this problem:

1. Detect customers spending at least n number of seconds around a register.
1. Estimate customer pose, verify hands are above the register table.
1. Add these identifications for human verification.
1. Train a video action recognition model to identify a target event.

#### Identify if cashier is right or left-handed (TODO)

Use tracks of pose estimation in order to make initial assumptions
about cashier handedness as frequency of usage of each hand.
If this approach leads to wrong detections, use extracted pose
features to train a simple machine learning model.
