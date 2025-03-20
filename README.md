# PCEBEAR: Pose Concept Bottleneck for Explainable Action Recognition

## Overview

PCEBEAR is a novel framework designed to provide interpretable video action recognition by leveraging pose-based concepts. Unlike traditional video action recognition models that rely on pixel-level importance or textual descriptions, PCEBEAR introduces a concept bottleneck that models both spatial joint configurations and temporal movement patterns, making it particularly suitable for explainable AI in the video domain.

The core of PCEBEAR is to represent actions through human pose sequences, capturing the fine-grained motion dynamics that are essential for understanding human activities. By using skeleton-based concepts, PCEBEAR overcomes the limitations of pixel-based supervoxel clustering and static textual concepts, providing a motion-aware explanation for action recognition.

## Key Features

- **Pose-Based Concepts:** PCEBEAR introduces two types of pose-based concepts: 
  - *Static Pose Concepts*: Capture spatial configurations at individual frames.
  - *Dynamic Pose Concepts*: Encode motion patterns across multiple frames.
  
- **Unsupervised Clustering:** We leverage an unsupervised clustering approach to group similar pose sequences and automatically discover meaningful pose-based concepts, removing the need for manual annotations.

- **Transparent Predictions:** The model provides clear and interpretable explanations by directly associating class predictions with pose-based concepts.

- **Flexibility and Efficiency:** PCEBEAR offers a flexible framework that works with any RGB input, eliminating the need for explicit skeleton inputs, and still provides interpretable action predictions.

## Datasets Used

- **KTH-5:** A refined version of the KTH dataset, focusing on five distinct action classes, excluding "jogging" due to overlap with other classes.
  
- **Penn Action:** Contains 2,326 video sequences covering 15 action classes with detailed human joint annotations, making it suitable for both action recognition and pose estimation tasks.
  
- **HAA49:** A subset of the HAA500 dataset, focusing on 49 human-centric action classes where pose information is crucial for distinguishing similar actions.

## How PCEBEAR Works

PCEBEAR works by first extracting pose sequences using a pre-trained pose estimator. Then, it uses an unsupervised clustering approach to discover pose-based concepts, which are subsequently used for video action recognition. These concepts can be classified into static or dynamic categories, capturing either spatial configurations or motion dynamics, respectively.

### Example Workflow:
1. **Pose Extraction:** Human pose sequences are extracted from the video frames using an off-the-shelf pose estimator.
2. **Concept Discovery:** Pose sequences are clustered into static and dynamic pose concepts using unsupervised clustering.
3. **Action Recognition:** These pose-based concepts are used to classify actions in the video.

<!-- ## Installation and Usage

### Requirements

- Python 3.6+
- PyTorch 1.8+
- Other dependencies listed in `requirements.txt` -->

<!-- ### Usage

```bash
# Clone the repository
git clone https://github.com/your-username/your-repository.git

# Install the required dependencies
cd your-repository
pip install -r requirements.txt

# To run the model
python run_model.py --dataset <dataset_name> --input <video_input_path> --output <output_path> -->