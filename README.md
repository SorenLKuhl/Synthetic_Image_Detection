# Synthetic_Image_Detection
 DTU course 02456: Deep Learning project of Johannes Poulsen (s204399) & Søren Kühl (s216161).

Note: Files contain paths to datasets outside of the repo as they are too large. So without changes, the script will not run properly.
 

## File structure
- models
   - cnn.py: Definition of our own CNN.
   - clip_REAL.py: Testing of CLIP zero-shot performance.
   - resnet18.py: Script to use ResNet18.
- config.ini: Path to directory of datasets.
- diffusion_faces.py: Tests a trained model on a dataset. Initially stable-diffusion.
- real_and_fake_faces.py: Trains and tests a model on the real_vs_fake dataset.
- submit.sh: Script to submit a job to DTU HPC.
- train_mixed_model.py: Script that trains and tests our CNN on mixed data.
