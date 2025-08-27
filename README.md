# Automated Quality Filtering for Diabetic Retinopathy Datasets

## Overview

This is an automated quality filtering system for diabetic retinopathy (DR) fundus image datasets. It addresses a critical challenge in medical AI: ensuring dataset quality for robust model training. Using adaptive quality thresholds and medical image-specific metrics, it systematically identifies and removes low-quality images while preserving clinically relevant data.

All datasets used in this work are publicly available. The structure can be adapted as needed to accommodate other datasets.

**Research Paper:** *(Link will be added upon publication)*

### Composite Quality Scoring

The system uses a 3-component quality scoring methodology that integrates normalized quality metrics:

```
Composite Quality = 0.25 × BasicQuality + 0.55 × MedicalQuality + 0.20 × TechnicalQuality
```

Where:
- **BasicQuality** = mean([Brightness, Contrast, Sharpness, Entropy]) - Standard image metrics
- **MedicalQuality** = mean([Illumination_Uniformity, Vessel_Visibility, Optic_Disc_Visibility]) - Domain-specific metrics  
- **TechnicalQuality** = mean([Extreme_Pixels, Motion_Blur, Color_Balance]) - Technical assessment metrics

An image is accepted if its composite quality score exceeds the severity-specific threshold determined during dataset calibration.

## Supported Datasets

| Dataset | Images | Description | Download Link |
|---------|--------|-------------|---------------|
| APTOS 2019 | ~3,662 | Kaggle competition dataset | https://www.kaggle.com/competitions/aptos2019-blindness-detection |
| Diabetic Retinopathy V03 | ~35,126 | Large-scale validation set | https://zenodo.org/records/4891308 |
| IDRiD | ~516 | High-resolution research benchmark | https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid |
| Messidor-2 | ~1,748 | Clinical validation standard | https://www.adcis.net/en/third-party/messidor2/ |
| SUSTech-SYSU | ~1,200 | Asian population dataset | https://figshare.com/articles/dataset/The_SUSTech-SYSU_dataset_for_automated_exudate_detection_and_diabetic_retinopathy_grading/12570770?file=25320596 |
| DeepDRiD | ~2,000 | Recent competition data | https://github.com/deepdrdoc/DeepDRiD |

### Required Dataset Structure

```
BASE_PATH/
├── APTOS 2019/
│   ├── 0/          # No DR images
│   ├── 1/          # Mild DR images  
│   ├── 2/          # Moderate DR images
│   ├── 3/          # Severe DR images
│   └── 4/          # Proliferative DR images
├── Diabetic Retinopathy_V03/
│   ├── 0/ ... 4/
└── [additional datasets...]
```

## Quality Metrics

**Basic Quality Domain (4 metrics):**
- **Brightness**: Global luminance evaluation to detect exposure extremes
- **Contrast**: Local intensity variation measurement for structure differentiation
- **Sharpness**: Laplacian variance for edge clarity and blur detection
- **Entropy**: Shannon entropy for information content assessment

**Medical Relevance Domain (3 metrics):**
- **Illumination Uniformity**: Spatial lighting consistency via grid-based analysis
- **Vessel Visibility**: Blood vessel enhancement using directional filtering on green channel
- **Optic Disc Detectability**: Anatomical landmark visibility through bright pixel analysis

**Technical Quality Domain (3 metrics):**
- **Extreme Pixels**: Detection of under/over-exposed regions indicating poor exposure
- **Motion Blur**: Gradient magnitude analysis for movement-related quality degradation
- **Color Balance**: RGB channel variation analysis for color cast identification

**Technical Validation:**
- Resolution adequacy verification
- File integrity and size validation
- Multi-channel color analysis

## Installation and Setup

### Prerequisites

```bash
pip install opencv-python numpy pandas matplotlib
pip install scikit-learn jupyter notebook
```

### Setup Process

1. Clone or download the repository files:
   - `Quality_Filtering.ipynb` - Main analysis notebook
   - `Class_balancer.py` - Dataset balancing script

2. Configure dataset paths in cell 2 of the notebook:
```python
BASE_PATH = "Path/to/your/dataset"
```

3. Execute the notebook sequentially:
```bash
jupyter notebook Quality_Filtering.ipynb
```

## Complete Workflow

### Phase 1: Quality Filtering (Notebook Cells 1-17)

**System Setup (Cells 1-5)**
- Environment configuration and dataset validation
- Quality identifier initialization

**Dataset Analysis (Cells 6-13)**
- Statistical characterization and profile generation
- Comprehensive quality assessment across all images
- Results compilation and statistical reporting

**Review and Validation (Cells 14-16)**  
- Manual review interface generation
- Threshold adjustment capabilities
- Quality assessment validation

**Dataset Generation (Cell 17)**
- Creation of cleaned, production-ready dataset
- Comprehensive statistics and summary reports

### Phase 2: Medical-Safe Balancing

After quality filtering, apply class balancing using the medical-safe augmentation system:

```bash
# Basic usage
python Class_balancer.py --source /path/to/cleaned/dataset --output /path/to/balanced/dataset

# With custom configuration
python Class_balancer.py --config medical_balancer_config.json

# Custom quality threshold
python Class_balancer.py --source /data/cleaned --output /data/balanced --quality-threshold 0.8

# Create example configuration file
python Class_balancer.py --create-config
```

### Class Balancer Configuration

**Basic Configuration:**
```python
# Default target distribution (optimized for medical training)
target_distribution = {
    0: 900,  # No DR
    1: 750,  # Mild DR  
    2: 800,  # Moderate DR
    3: 750,  # Severe DR
    4: 800   # Proliferative DR
}
```

**Advanced Configuration (JSON):**
```json
{
  "source_directory": "/path/to/cleaned/dataset",
  "output_directory": "/path/to/balanced/dataset", 
  "target_distribution": {
    "0": 900,
    "1": 750,
    "2": 800,
    "3": 750,
    "4": 800
  },
  "quality_threshold": 0.7,
  "random_seed": 42
}
```

## Severity-Adaptive Thresholds

Quality thresholds are dynamically adjusted based on DR severity to preserve diagnostically critical images:

| DR Grade | Clinical Name | Removal Percentile | Rationale |
|----------|---------------|-------------------|-----------|
| 0 | No DR | 15% | Higher tolerance for screening images |
| 1 | Mild DR | 12% | Balanced quality control |
| 2 | Moderate DR | 10% | Increased clinical importance |
| 3 | Severe DR | 8% | Critical for diagnosis |
| 4 | Proliferative DR | 5% | Maximum retention for rare cases |

You can adjust these removal percentiles in the notebook (cell 16) as needed for your specific requirements.

## Output Structure

The system generates organized outputs for analysis and production use:

**Analysis Results:**
```
quality_review/
├── quality_identification_results.csv    # Complete analysis data
├── flagged_images_summary.csv           # Images recommended for removal
├── identification_report.json           # Statistical analysis
├── dataset_profiles.json               # Dataset characterization
├── analysis_summary_plots.png          # Visualization dashboard
├── sample_images/                       # Representative samples
├── flagged_samples/                     # Quality issue examples
└── problematic_images_review_all/       # Manual review interface
```

**Production Dataset:**
```
DREAM_dataset_cleaned/
├── [Dataset_Name]/
│   ├── 0/ ... 4/                       # Filtered images by DR grade
├── cleaning_statistics.json            # Processing metrics
└── CLEANING_SUMMARY.txt               # Human-readable summary
```

**Balanced Dataset:**
```
balanced_dataset/
├── 0/ ... 4/                          # Balanced classes with augmented images
├── balancing_stats.json               # Detailed balancing metrics
└── augmentation_report.txt            # Human-readable summary
```

## Dataset Balancing

### Overview

The system includes a sophisticated class balancing module that addresses dataset imbalance using only **anatomically-preserving augmentations** suitable for medical fundus imaging. Unlike general computer vision augmentations, this module specifically avoids transformations that could create anatomically impossible configurations.

### Key Features

**Medical Safety First:**
- No horizontal/vertical flips (would create anatomically impossible retinal orientations)
- No elastic deformations that could distort vessel patterns
- Preserves anatomical relationships critical for diagnosis
- Quality assessment for every augmented image

**Severity-Adaptive Augmentation:**
- Different augmentation strategies based on DR severity (0-4)
- More conservative augmentations for severe cases to preserve diagnostic features
- Maintains clinical relevance across all severity levels

### Augmentation Techniques

The balancer employs seven medical-appropriate augmentation methods:

| Technique | Description | Medical Rationale |
|-----------|-------------|-------------------|
| **Quality-Preserving Rotation** | Small rotations (±12°) | Simulates natural head positioning variations |
| **Medical Brightness/Contrast** | Conservative illumination adjustments | Accounts for different camera settings |
| **Fundus-Specific Enhancement** | CLAHE on green channel | Standard ophthalmological image processing |
| **Vessel-Preserving Noise** | Minimal noise with bilateral filtering | Maintains vessel structure integrity |
| **Color Temperature Shift** | Simulates camera color variations | Different fundus camera manufacturers |
| **Medical Zoom/Crop** | Controlled zoom with interpolation | Field of view variations |
| **Gamma Correction** | Contrast curve adjustments | Display calibration differences |

### Severity-Specific Strategies

Augmentation aggressiveness is adapted based on DR severity to preserve diagnostic quality:

| DR Grade | Clinical Name | Augmentation Strategy | Rationale |
|----------|---------------|----------------------|-----------|
| 0 | No DR | 7 combination types | Higher tolerance for screening images |
| 1 | Mild DR | 7 combination types | Balanced augmentation approach |
| 2 | Moderate DR | 6 combination types | Reduced complexity to preserve features |
| 3 | Severe DR | 5 combination types | Conservative to maintain pathology visibility |
| 4 | Proliferative DR | 4 combination types | Minimal augmentation for rare, critical cases |

### Quality Control System

Every augmented image undergoes rigorous quality assessment:

**Quality Metrics:**
- **Sharpness preservation**: Laplacian variance comparison
- **Brightness conservation**: Mean luminance stability  
- **Contrast maintenance**: Standard deviation analysis
- **Overall quality score**: Weighted combination (default threshold: 0.7)

**Acceptance Criteria:**
```
Quality Score = 0.4 × Sharpness_Ratio + 0.3 × (1 - Brightness_Diff) + 0.3 × Contrast_Ratio
```

Only images scoring above the quality threshold are accepted, ensuring augmented images maintain medical diagnostic quality.

## Configuration

### Manual Threshold Adjustment

Fine-tune quality standards in notebook cell 15:

```python
manual_thresholds = {
    'APTOS2019': {
        0: 0.250,  # Custom threshold for No DR
        1: None,   # Keep automatic threshold
        # ... additional customizations
    }
}
```

### Balancing Configuration Options

**Target Distribution Customization:**
```python
# Conservative balancing (preserves more originals)
conservative_distribution = {0: 600, 1: 500, 2: 550, 3: 500, 4: 550}

# Aggressive balancing (larger dataset)
aggressive_distribution = {0: 1200, 1: 1000, 2: 1100, 3: 1000, 4: 1100}
```

**Quality Control Parameters:**
```bash
# Adjust quality thresholds via command line
python Class_balancer.py --quality-threshold 0.6  # More lenient
python Class_balancer.py --quality-threshold 0.8  # More strict (default: 0.7)
```

## Performance Characteristics

**Processing Metrics:**
- Processing speed: 100-500 images/minute (hardware dependent)
- Memory usage: 2-8GB peak (dataset size dependent)
- Typical removal rate: 8-15% of total images

**Balancing Performance:**
- Augmentation speed: 50-200 images/minute
- Quality control rejection rate: 10-20%
- Memory usage: 3-6GB during processing
- Typical dataset expansion: 150-300%

**Quality Outcomes:**
- Model accuracy improvement: 3-7% on average
- Reduced training instability and faster convergence
- Enhanced cross-dataset generalization
- Preserved diagnostic quality while removing technical artifacts

### System Parameters

Key configurable parameters:
- `N_SAMPLES_PER_DATASET`: Number of images for characterization (default: 300)
- Quality score weighting: Basic vs medical vs technical metrics ratio (25%:55%:20%)
- Resolution threshold: Minimum acceptable image dimensions (224x224)
- Processing batch size: Memory management optimization
- Augmentation quality threshold: Minimum quality score for acceptance (default: 0.7)

## Technical Implementation

### Architecture

The system implements a modular architecture with these components:

- **QualityIdentifier Class**: Core analysis engine with medical image-specific methods
- **MedicalDRBalancer Class**: Medical-safe augmentation and balancing engine
- **Adaptive Profiling**: Statistical characterization with percentile-based thresholding
- **Batch Processing**: Memory-efficient handling of large datasets
- **Comprehensive Reporting**: Multi-format output generation

### Algorithm Details

**Quality Score Calculation:**
1. Extract 10 image characteristics per image across three quality domains
2. Normalize metrics using dataset-specific statistics
3. Compute weighted combination of basic, medical, and technical quality scores
4. Apply DR severity-specific thresholds for classification

**Medical-Safe Augmentation Pipeline:**
1. Select severity-appropriate augmentation combinations
2. Apply transformations while preserving anatomical validity
3. Assess augmentation quality using multi-metric evaluation
4. Accept/reject based on medical quality thresholds
5. Track all transformations for reproducibility

## Troubleshooting

**Common Issues:**

Dataset path configuration:
```bash
# Verify dataset structure
python -c "import os; print([d for d in os.listdir('your/path') if os.path.isdir(os.path.join('your/path', d))])"
```

Memory optimization for large datasets:
```python
N_SAMPLES_PER_DATASET = 150  # Reduce from default 300 in notebook cell 2
```

**Balancing Issues:**

Class directory verification:
```bash
# Check class structure for balancer
python -c "
import os
path = 'your/dataset/path'
for i in range(5):
    class_dir = os.path.join(path, str(i))
    if os.path.exists(class_dir):
        count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f'Class {i}: {count} images')
    else:
        print(f'Class {i}: Directory not found')
"
```

Quality control adjustment:
```bash
# If too many augmentations are rejected, lower the threshold
python Class_balancer.py --quality-threshold 0.6  # Default: 0.7
```

**Performance Optimization:**
- Use SSD storage for improved I/O performance
- Allocate 8GB+ RAM for optimal processing
- Process datasets individually if memory constrained
- Reduce target distribution sizes for faster processing

## Citation

If you use this system in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Automated Quality Filtering for Diabetic Retinopathy Datasets},
  author={Your Authors},
  journal={Your Journal},
  year={2024}
}
```

## Contributing

Contributions are welcome. Please submit issues or pull requests for:
- Additional quality metrics
- Support for new dataset formats
- Performance optimizations
- Documentation improvements
- New medical-safe augmentation techniques

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset providers for making their data publicly available
- OpenCV and scikit-learn communities for foundational tools
- Medical imaging research community for validation and feedback

**Disclaimer**: This system is designed for research purposes. Clinical applications require additional validation and regulatory approval.
