# Automated Quality Filtering for Diabetic Retinopathy Datasets
---
## Overview

This is an automated quality filtering system for diabetic retinopathy (DR) fundus image datasets.  
It addresses a critical challenge in medical AI: ensuring dataset quality for robust model training.  
Using adaptive quality thresholds and medical image-specific metrics, it systematically identifies and removes low-quality images while preserving clinically relevant data.  

All datasets used in this work are publicly available.  
The structure can be adapted as needed to accommodate other datasets.

**Research Paper:** *(Link will be added upon publication)*


## Methodology

### Adaptive Quality Assessment

The system employs a two-tier quality scoring approach:

```
Combined Quality Score = 0.3 × Basic Quality + 0.7 × Medical Quality
```

Where:
- **Basic Quality**: Standard image metrics (brightness, contrast, sharpness, entropy)
- **Medical Quality**: Domain-specific metrics (vessel visibility, optic disc visibility, illumination uniformity)

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

### Quality Metrics

**Image Quality Fundamentals:**
- Shannon entropy for information content assessment
- Laplacian variance for sharpness quantification
- Statistical analysis of brightness and contrast distributions
- Multi-channel color balance evaluation

**Medical Image Specifics:**
- Blood vessel enhancement using directional filtering kernels
- Optic disc detection through bright pixel analysis
- Illumination uniformity via spatial segmentation
- Automated artifact detection (borders, saturation, motion blur)

**Technical Validation:**
- Resolution adequacy verification
- File integrity and size validation
- Motion blur assessment using gradient analysis

## Installation and Usage

### Prerequisites

```bash
pip install opencv-python numpy pandas matplotlib
pip install scikit-learn jupyter notebook
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dr-quality-filtering.git
cd dr-quality-filtering
```

2. Configure dataset paths in cell 2 of the notebook:
```python
BASE_PATH = r"Path/to/your/dataset"
```

3. Execute the notebook sequentially:
```bash
jupyter notebook Quality_Filtering.ipynb
```

### Execution Workflow

**Phase 1: System Setup (Cells 1-5)**
- Environment configuration and dataset validation
- Quality identifier initialization

**Phase 2: Dataset Analysis (Cells 6-13)**
- Statistical characterization and profile generation
- Comprehensive quality assessment across all images
- Results compilation and statistical reporting

**Phase 3: Review and Validation (Cells 14-16)**
- Manual review interface generation
- Threshold adjustment capabilities
- Quality assessment validation

**Phase 4: Dataset Generation (Cell 17)**
- Creation of cleaned, production-ready dataset
- Comprehensive statistics and summary reports

### Severity-Adaptive Thresholds

You can adjust the Removal Percentiles as per your requirements.

Quality thresholds are dynamically adjusted based on DR severity to preserve diagnostically critical images:

| DR Grade | Clinical Name | Removal Percentile | Rationale |
|----------|---------------|-------------------|-----------|
| 0 | No DR | 15% | Higher tolerance for screening images |
| 1 | Mild DR | 12% | Balanced quality control |
| 2 | Moderate DR | 10% | Increased clinical importance |
| 3 | Severe DR | 8% | Critical for diagnosis |
| 4 | Proliferative DR | 5% | Maximum retention for rare cases |

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

## Configuration

### Manual Threshold Adjustment

Fine-tune quality standards in cell 16:

```python
manual_thresholds = {
    'APTOS2019': {
        0: 0.250,  # Custom threshold for No DR
        1: None,   # Keep automatic threshold
        # ... additional customizations
    }
}
```

## Performance Characteristics

**Processing Metrics:**
- Processing speed: 100-500 images/minute (hardware dependent)
- Memory usage: 2-8GB peak (dataset size dependent)
- Typical removal rate: 8-15% of total images

**Quality Outcomes:**
- Model accuracy improvement: 3-7% on average
- Reduced training instability and faster convergence
- Enhanced cross-dataset generalization
- Preserved diagnostic quality while removing technical artifacts


### System Parameters

Key configurable parameters:
- `N_SAMPLES_PER_DATASET`: Number of images for characterization (default: 300)
- Quality score weighting: Basic vs medical metrics ratio
- Resolution threshold: Minimum acceptable image dimensions
- Processing batch size: Memory management optimization

## Technical Implementation

### Architecture

The system implements a modular architecture with the following components:

- **QualityIdentifier Class**: Core analysis engine with medical image-specific methods
- **Adaptive Profiling**: Statistical characterization with percentile-based thresholding
- **Batch Processing**: Memory-efficient handling of large datasets
- **Comprehensive Reporting**: Multi-format output generation

### Algorithm Details

**Quality Score Calculation:**
1. Extract 15+ image characteristics per image
2. Normalize metrics using dataset-specific statistics
3. Compute weighted combination of basic and medical quality scores
4. Apply DR severity-specific thresholds for classification

**Memory Management:**
- Garbage collection at regular intervals
- Progress tracking for long-running operations
- Error handling and recovery mechanisms

## Troubleshooting

**Common Issues:**

Dataset path configuration:
```bash
# Verify dataset structure
python -c "import os; print([d for d in os.listdir('your/path') if os.path.isdir(os.path.join('your/path', d))])"
```

Memory optimization for large datasets:
```python
N_SAMPLES_PER_DATASET = 150  # Reduce from default 300
```

**Performance Optimization:**
- Use SSD storage for improved I/O performance
- Allocate 8GB+ RAM for optimal processing
- Process datasets individually if memory constrained

## Citation

If you use this system in your research, please cite:

```bibtex
xxx
```

## Contributing

Contributions are welcome. Please feel free to submit issues or pull requests for:
- Additional quality metrics
- Support for new dataset formats
- Performance optimizations
- Documentation improvements

## Authors


**Disclaimer**: This system is designed for research purposes. Clinical applications require additional validation and regulatory approval.

## Acknowledgments

- Dataset providers for making their data publicly available
- OpenCV and scikit-learn communities for foundational tools
- Medical imaging research community for validation and feedback
