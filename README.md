# Unsupervised-Optical-Flow

We used the training pipeline from <a href="https://github.com/nianticlabs/monodepth2"> monodepth2 </a>
 
Using the pose estimator module, we can compute the displacement at a pixel level from source to target. We extracted a rigid flow field, which is valid only for the static objects in the scene and captures the car's motion relative to the environment but fails to capture objects in motion.

```shell
mkdir models
```
A pretrained model for the UPB dataset is available <a href="https://drive.google.com/drive/folders/18kTR4PaRlQIeEFJ2gNkiXYnFcTfyrRNH?usp=sharing"> here </a>. 
Copy all the files into the "models" directory.
 
 
