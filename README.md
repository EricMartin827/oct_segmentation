# Ret-U-Net for OCT RNFL Segmentation

This project develops a deep learning model and pipeline to segment the RNFL layer of retina OCT images for rats.

This study is based on the course project of Georgia Tech CS7643 Deep Learning. A summary of the study can be found in the PDF report. It is an ongoing project with future improvement.

In this repo you will find resources to:
* Preprocess OCT data
* Run a Monai U-Net model for three-class retina segmentation (backgroudn, RNFL layer, other retina layers) with different configurations
* Post-analysis of the segmentation results


## Data 

Data contains 288 images from 60 rats acquired at the Feola Lab at Emory University. OCT scans were collected by Bioptigen 4300 (Leica Microsystems). Images were then registered and averaged leading to a sample size of 4 scans per eye. Animals had unilateral RGC injury by ocular hypertension (n=38, 3-10 month Brown Norway rats) or optic nerve crush (n=12, 9-10 month Long-Evans rats); contralateral eyes serving as controls. Scans were acquired at 4- and 8- (hypertensive cohort) or 12- (crush cohort) weeks after injury. We consider scans from the same animal acquired at each time point as independent.
These scans were manually annotated by trained human annotators to create the ground truths. 

## Scripts
`configs` Configuration files

`data` Preprocessed OCT data (train/val/test) and augmented data.

`notebooks/post_training_evaluation` Jupyter notebooks that contain post-training analysis of segmentation results

`oct` main functions used to train and evaluate the U-Net.
