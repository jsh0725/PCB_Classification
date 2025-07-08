## Project Structure

- `DeepPCB_split/`  
  Preprocessed dataset used for model training and validation. The original DeepPCB dataset has been reorganized for this project.

- `Binary_classification.ipynb`  
  Contains implementations of five different binary classification models to distinguish between normal and defective PCB images.

- `Multi_Label.ipynb`  
  Handles multi-label defect localization. Uses LabelMe for mask-based preprocessing, enabling the model to detect and localize defect types with red rectangular bounding boxes.

- `README.md`  
  Project documentation.

## Key Features

- Binary classification of PCB images (normal vs. defective)
- Defect type detection and localization with bounding boxes
- Custom dataset preprocessing based on DeepPCB
- LabelMe-based annotation and mask generation