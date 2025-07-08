## Project Structure
   This project was conducted as part of a collaborative assignment.

- `DeepPCB_split/`  
  Preprocessed dataset used for model training and validation. The original DeepPCB dataset has been reorganized for this project.
  
  predict.py is an early-stage Python script that contains five binary classification models. It stores the results of each model's predictions, which are then visualized and analyzed using the visualize_all_models.py file.

- `Binary_classification.ipynb`(model was developed collaboratively)  
  Contains implementations of five different binary classification models to distinguish between normal and defective PCB images.

- `Multi_Label.ipynb` (Individually developed.) 
  Handles multi-label defect localization. Uses LabelMe for mask-based preprocessing, enabling the model to detect and localize defect types with red rectangular bounding boxes.

- `KerasCV.ipynb` (This part was developed by a collaborator.) 
  Implements object detection using KerasCV's RetinaNet model with a ResNet50 backbone. Trains on PCB images annotated in Pascal VOC format to detect and classify multiple defect types using bounding boxes with class labels and confidence scores.

- `README.md`  
  Project documentation.

## Key Features

- Binary classification of PCB images (normal vs. defective)
- Defect type detection and localization with bounding boxes
- Custom dataset preprocessing based on DeepPCB
- LabelMe-based annotation and mask generation