Search for:
Search

[Skip to content](https://lila.science/datasets/caltech-camera-traps#content)

Overview

This data set contains 243,100 images from 140 camera locations in the Southwestern United States, with labels for 21 animal categories (plus empty), primarily at the species level (for example, the most common labels are opossum, raccoon, and coyote), and approximately 66,000 bounding box annotations. Approximately 70% of images are labeled as empty.

More information about this data set is available [here](https://beerys.github.io/CaltechCameraTraps/).

Citation, license, and contact information

If you use this data set, please cite the associated manuscript:

Sara Beery, Grant Van Horn, Pietro Perona. Recognition in Terra Incognita. Proceedings of the 15th European Conference on Computer Vision (ECCV 2018). ( [bibtex](http://lila.science/wp-content/uploads/2018/11/BeeryHP18.bib))

This data set is released under the [Community Data License Agreement (permissive variant)](https://cdla.io/permissive-1-0/).

For questions about this data set, contact [caltechcameratraps@gmail.com](mailto:caltechcameratraps@gmail.com).

Data format

Annotations are provided in [COCO Camera Traps](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/data_management/README.md#coco-camera-traps-format) .json format.

We have also divided locations (i.e., cameras) into training and validation splits to allow for consistent benchmarking on this data set. The file describing this split specifies a train/val split for all locations in the data set, and also provides the train/val split used in the ECCV paper listed above. The “eccv\_train” split here corresponds to the “train” locations and all “cis” locations in the ECCV paper; the “eccv\_val” split here corresponds to all “trans” locations in the ECCV paper.

For information about mapping this dataset’s categories to a common taxonomy, see [this page](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/).

Downloading the data

Images are available in the following cloud storage folders:

- gs://public-datasets-lila/caltech-unzipped/cct\_images (GCP)
- s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/caltech-unzipped/cct\_images (AWS)
- https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct\_images (Azure)

A link to a zipfile is also provided below, but – whether you want the whole data set, a specific folder, or a subset of the data (e.g. images for one species) – we recommend checking out our [guidelines for accessing images without using giant zipfiles](http://lila.science/image-access).

- Images (105GB) ( [GCP link](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/cct_images.tar.gz)) ( [Azure link](https://lilawildlife.blob.core.windows.net/lila-wildlife/caltechcameratraps/cct_images.tar.gz)) ( [AWS link](http://us-west-2.opendata.source.coop.s3.amazonaws.com/agentmorris/lila-wildlife/caltechcameratraps/cct_images.tar.gz))

Metadata download links:

- [Image-level annotations](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip) (9MB)
- [Bounding box annotations](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_bboxes_20200316.json) (35MB)
- [Recommended train/val splits](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/CaltechCameraTrapsSplits_v0.json) (4KB)

Having trouble downloading? Check out our [FAQ](http://lila.science/faq#downloadtips).

CCT20 Benchmark subset

The initial publication of this data set in [Beery et al. 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Beery_Recognition_in_Terra_ECCV_2018_paper.html) proposed a specific subset of Caltech Camera Traps data for benchmarking. Images in this benchmark dataset (CCT20) have been downsized to a maximum of 1024 pixels on a side.

The CCT20 benchmark set is available here:

[Benchmark images](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz) (6GB)

[Metadata files for train/val/cis/trans splits](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz) (3MB)

Other useful links

[MegaDetector](https://github.com/agentmorris/MegaDetector/blob/main/megadetector.md) results for all camera trap datasets on LILA are available [here](https://lila.science/megadetector-results-for-camera-trap-datasets/).

Information about mapping camera trap datasets to a common taxonomy is available [here](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/).

![](http://lila.science/wp-content/uploads/2018/10/CaltechCameraTraps_web.jpg)

Posted by [Dan Morris](http://dmorris.net/).