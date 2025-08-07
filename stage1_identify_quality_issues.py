import os
import json
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from datetime import datetime
import logging
from collections import Counter
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QualityIdentifier:
    def __init__(self, output_dir='quality_review', random_seed=42):
        self.output_dir = output_dir
        self.dataset_profiles = {}
        self.identification_results = []
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/sample_images', exist_ok=True)
        os.makedirs(f'{output_dir}/flagged_samples', exist_ok=True)
        
        logger.info(f"Quality identification output directory: {output_dir}")
    
    def characterize_dataset(self, dataset_path, dataset_name, n_samples=300):
        """Characterize dataset properties for adaptive thresholding"""
        logger.info(f"Characterizing {dataset_name}...")
        
        # Sample images strategically
        sample_images = self.sample_images_strategically(dataset_path, n_samples)
        
        if not sample_images:
            logger.warning(f"No images found in {dataset_path}")
            return None
        
        logger.info(f"  Analyzing {len(sample_images)} sample images...")
        
        # Analyze characteristics
        characteristics = []
        for i, img_path in enumerate(sample_images):
            if i % 50 == 0:
                logger.info(f"    Processing sample {i+1}/{len(sample_images)}")
            
            char = self.analyze_single_image(img_path)
            if char:
                characteristics.append(char)
            
            # Memory management
            if i % 100 == 0:
                gc.collect()
        
        if not characteristics:
            logger.warning(f"No valid characteristics extracted from {dataset_name}")
            return None
        
        # Calculate dataset profile
        profile = self.calculate_dataset_profile(dataset_name, characteristics)
        
        # Save sample images for review
        self.save_sample_images(sample_images[:20], dataset_name)
        
        return profile
    
    def sample_images_strategically(self, dataset_path, n_samples):
        """Sample images from all DR classes proportionally"""
        all_images = []
        class_images = {0: [], 1: [], 2: [], 3: [], 4: []}
        
        logger.info(f"  Scanning directory structure...")
        
        # Collect all images by class
        image_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.JPG', '.JPEG', '.PNG')
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_path = os.path.join(root, file)
                    dr_class = self.extract_dr_label_from_path(image_path)
                    if dr_class is not None:
                        class_images[dr_class].append(image_path)
        
        # Log class distribution
        for dr_class, images in class_images.items():
            logger.info(f"    DR Class {dr_class}: {len(images)} images")
        
        # Sample proportionally from each class
        sampled_images = []
        total_available = sum(len(images) for images in class_images.values())
        
        if total_available == 0:
            logger.warning("No images found with valid DR labels")
            return []
        
        samples_per_class = max(1, n_samples // 5)
        
        for dr_class, images in class_images.items():
            if images:
                n_class_samples = min(samples_per_class, len(images))
                sampled = np.random.choice(images, n_class_samples, replace=False)
                sampled_images.extend(sampled)
        
        logger.info(f"  Selected {len(sampled_images)} stratified samples")
        return sampled_images
    
    def extract_dr_label_from_path(self, image_path):
        """Extract DR severity from path - CUSTOMIZE THIS FOR YOUR DATASETS"""
        parts = image_path.split(os.sep)
        
        # Method 1: Check for folder names that are DR classes
        for part in parts:
            if part.isdigit() and int(part) in [0, 1, 2, 3, 4]:
                return int(part)
        
        # Method 2: Check for common DR folder naming patterns
        dr_patterns = {
            'no_dr': 0, 'normal': 0, 'grade_0': 0, 'class_0': 0,
            'mild': 1, 'grade_1': 1, 'class_1': 1,
            'moderate': 2, 'grade_2': 2, 'class_2': 2,
            'severe': 3, 'grade_3': 3, 'class_3': 3,
            'proliferative': 4, 'grade_4': 4, 'class_4': 4
        }
        
        for part in parts:
            part_lower = part.lower()
            if part_lower in dr_patterns:
                return dr_patterns[part_lower]
        
        # Method 3: Extract from filename (customize based on your naming convention)
        filename = os.path.basename(image_path).lower()
        
        # Example patterns - adjust for your datasets:
        # If filenames like "image_grade_2.jpg" or "patient_001_severity_3.png"
        if '_grade_' in filename:
            try:
                grade_part = filename.split('_grade_')[1]
                grade = int(grade_part.split('_')[0].split('.')[0])
                if grade in [0, 1, 2, 3, 4]:
                    return grade
            except (ValueError, IndexError):
                pass
        
        if '_severity_' in filename:
            try:
                sev_part = filename.split('_severity_')[1]
                severity = int(sev_part.split('_')[0].split('.')[0])
                if severity in [0, 1, 2, 3, 4]:
                    return severity
            except (ValueError, IndexError):
                pass
        
        # If no pattern matches, return None
        logger.warning(f"Could not extract DR label from: {image_path}")
        return None
    
    def analyze_single_image(self, image_path):
        """Analyze single image characteristics"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return None
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape[:2]
            
            # Check for minimum size requirements
            if h < 50 or w < 50:
                logger.warning(f"Image too small: {image_path} ({w}x{h})")
                return None
            
            characteristics = {
                'image_path': image_path,
                'filename': os.path.basename(image_path),
                'resolution': (w, h),
                'file_size_mb': os.path.getsize(image_path) / (1024*1024),
                
                # Basic quality metrics
                'brightness': float(np.mean(image)),
                'contrast': float(np.std(image)),
                'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                'entropy': self.calculate_entropy(gray),
                
                # Color analysis (important for different cameras/ethnicities)
                'red_mean': float(np.mean(image[:,:,2])),
                'green_mean': float(np.mean(image[:,:,1])),
                'blue_mean': float(np.mean(image[:,:,0])),
                'color_balance': float(np.std([np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2])])),
                
                # Medical structure visibility
                'vessel_visibility': self.assess_vessel_visibility(image),
                'optic_disc_visibility': self.assess_optic_disc_visibility(image),
                'illumination_uniformity': self.assess_illumination_uniformity(gray),
                
                # Potential artifact detection
                'has_black_borders': self.detect_black_borders(gray),
                'extreme_brightness_pixels': self.detect_extreme_pixels(gray),
                'motion_blur_score': self.assess_motion_blur(gray)
            }
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            return None
    
    def calculate_entropy(self, image):
        """Calculate Shannon entropy"""
        try:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist[hist > 0]
            if len(hist) == 0:
                return 0.0
            prob = hist / hist.sum()
            return float(-np.sum(prob * np.log2(prob)))
        except Exception:
            return 0.0
    
    def assess_vessel_visibility(self, image):
        """Quick vessel visibility assessment"""
        try:
            green = image[:, :, 1]
            # Simple vessel enhancement filter
            kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
            filtered = cv2.filter2D(green, cv2.CV_32F, kernel)
            vessel_pixels = np.sum(filtered > np.percentile(filtered, 98))
            total_pixels = filtered.shape[0] * filtered.shape[1]
            return float(vessel_pixels / max(total_pixels, 1))
        except Exception:
            return 0.0
    
    def assess_optic_disc_visibility(self, image):
        """Quick optic disc visibility assessment"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            bright_pixels = np.sum(gray > np.percentile(gray, 95))
            total_pixels = gray.shape[0] * gray.shape[1]
            return float(bright_pixels / max(total_pixels, 1))
        except Exception:
            return 0.0
    
    def assess_illumination_uniformity(self, gray):
        """Assess illumination uniformity across image"""
        try:
            h, w = gray.shape
            regions = []
            
            # Divide into 3x3 grid
            for i in range(3):
                for j in range(3):
                    start_y, end_y = i * h // 3, (i + 1) * h // 3
                    start_x, end_x = j * w // 3, (j + 1) * w // 3
                    region = gray[start_y:end_y, start_x:end_x]
                    regions.append(np.mean(region))
            
            mean_intensity = np.mean(regions)
            if mean_intensity > 0:
                cv_score = np.std(regions) / mean_intensity
                return float(max(0, 1 - cv_score))
            return 0.0
        except Exception:
            return 0.0
    
    def detect_black_borders(self, gray):
        """Detect presence of black borders"""
        try:
            h, w = gray.shape
            border_thickness = min(50, h//10, w//10)
            
            if border_thickness < 1:
                return False
            
            # Check borders
            top_border = np.mean(gray[:border_thickness, :])
            bottom_border = np.mean(gray[-border_thickness:, :])
            left_border = np.mean(gray[:, :border_thickness])
            right_border = np.mean(gray[:, -border_thickness:])
            
            # Consider it has black borders if any border is very dark
            black_threshold = 30
            return any(border < black_threshold for border in [top_border, bottom_border, left_border, right_border])
        except Exception:
            return False
    
    def detect_extreme_pixels(self, gray):
        """Detect percentage of extremely bright or dark pixels"""
        try:
            very_dark = np.sum(gray < 10)
            very_bright = np.sum(gray > 245)
            total_pixels = gray.shape[0] * gray.shape[1]
            return float((very_dark + very_bright) / max(total_pixels, 1))
        except Exception:
            return 0.0
    
    def assess_motion_blur(self, gray):
        """Assess motion blur using gradient analysis"""
        try:
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            return float(np.mean(magnitude))
        except Exception:
            return 0.0
    
    def calculate_dataset_profile(self, dataset_name, characteristics):
        """Calculate comprehensive dataset profile"""
        profile = {
            'dataset_name': dataset_name,
            'analysis_date': datetime.now().isoformat(),
            'n_samples_analyzed': len(characteristics),
            'characteristics_stats': {},
            'adaptive_thresholds': {}
        }
        
        # Calculate statistics for each characteristic
        numeric_keys = [key for key in characteristics[0].keys() 
                       if key not in ['image_path', 'filename', 'resolution']]
        
        for key in numeric_keys:
            values = [char[key] for char in characteristics if char[key] is not None]
            if values:
                profile['characteristics_stats'][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'percentiles': {
                        '5': float(np.percentile(values, 5)),
                        '10': float(np.percentile(values, 10)),
                        '25': float(np.percentile(values, 25)),
                        '50': float(np.percentile(values, 50)),
                        '75': float(np.percentile(values, 75)),
                        '90': float(np.percentile(values, 90)),
                        '95': float(np.percentile(values, 95))
                    }
                }
        
        # Calculate adaptive removal thresholds (percentile-based)
        # These determine what percentage of each DR class to potentially remove
        removal_percentiles = {
            0: 15,  # No DR - consider removing worst 15%
            1: 12,  # Mild DR - consider removing worst 12%
            2: 10,  # Moderate DR - consider removing worst 10%
            3: 8,   # Severe DR - consider removing worst 8%
            4: 5    # Proliferative DR - consider removing worst 5%
        }
        
        # Calculate combined quality scores for threshold calculation
        quality_scores = []
        for char in characteristics:
            # Normalize metrics (simplified version)
            brightness_norm = min(1.0, max(0.0, char['brightness'] / 255.0))
            contrast_norm = min(1.0, max(0.0, char['contrast'] / 100.0))
            sharpness_norm = min(1.0, max(0.0, char['sharpness'] / 1000.0))
            entropy_norm = min(1.0, max(0.0, char['entropy'] / 8.0))
            
            basic_quality = np.mean([brightness_norm, contrast_norm, sharpness_norm, entropy_norm])
            
            medical_quality = np.mean([
                char['illumination_uniformity'],
                min(1.0, char['vessel_visibility'] * 10),
                min(1.0, char['optic_disc_visibility'] * 10)
            ])
            
            combined_quality = 0.3 * basic_quality + 0.7 * medical_quality
            quality_scores.append(combined_quality)
        
        # Set percentile-based thresholds
        for dr_severity, percentile in removal_percentiles.items():
            threshold = np.percentile(quality_scores, percentile) if quality_scores else 0.3
            profile['adaptive_thresholds'][dr_severity] = float(threshold)
        
        return profile
    
    def save_sample_images(self, sample_paths, dataset_name):
        """Save sample images for visual review"""
        sample_dir = f'{self.output_dir}/sample_images/{dataset_name}'
        os.makedirs(sample_dir, exist_ok=True)
        
        for i, img_path in enumerate(sample_paths):
            try:
                dst_path = f'{sample_dir}/sample_{i:02d}_{os.path.basename(img_path)}'
                shutil.copy2(img_path, dst_path)
            except Exception as e:
                logger.error(f"Error copying sample {img_path}: {e}")
    
    def identify_quality_issues(self, dataset_path, dataset_name, profile):
        """Identify images with quality issues"""
        logger.info(f"Identifying quality issues in {dataset_name}...")
        
        results = []
        processed_count = 0
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.JPG', '.JPEG', '.PNG')
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_path = os.path.join(root, file)
                    dr_severity = self.extract_dr_label_from_path(image_path)
                    
                    if dr_severity is None:
                        continue
                    
                    # Analyze image
                    characteristics = self.analyze_single_image(image_path)
                    if not characteristics:
                        continue
                    
                    # Calculate quality scores
                    quality_assessment = self.assess_image_quality(characteristics, profile, dr_severity)
                    
                    # Create result record
                    result = {
                        'dataset': dataset_name,
                        'image_path': image_path,
                        'filename': file,
                        'dr_severity': dr_severity,
                        'overall_quality_score': quality_assessment['overall_score'],
                        'threshold_used': quality_assessment['threshold'],
                        'recommended_action': quality_assessment['action'],
                        'removal_reasons': quality_assessment['reasons'],
                        'confidence': quality_assessment['confidence']
                    }
                    
                    # Add detailed scores
                    result.update(characteristics)
                    result.update({f'normalized_{k}': v for k, v in quality_assessment['normalized_scores'].items()})
                    
                    results.append(result)
                    processed_count += 1
                    
                    if processed_count % 1000 == 0:
                        logger.info(f"  Processed {processed_count} images...")
                        gc.collect()  # Memory management
        
        logger.info(f"  Completed: {processed_count} images analyzed")
        return results
    
    def assess_image_quality(self, characteristics, profile, dr_severity):
        """Assess single image quality against dataset profile"""
        
        # Normalize characteristics relative to dataset
        normalized_scores = {}
        char_stats = profile['characteristics_stats']
        
        # Normalize key metrics
        key_metrics = ['brightness', 'contrast', 'sharpness', 'entropy', 
                      'illumination_uniformity', 'vessel_visibility', 'optic_disc_visibility']
        
        for metric in key_metrics:
            if metric in char_stats and metric in characteristics:
                stats = char_stats[metric]
                value = characteristics[metric]
                
                # Z-score normalization, then convert to 0-1 scale
                if stats['std'] > 0:
                    z_score = (value - stats['mean']) / stats['std']
                    normalized = max(0, min(1, (z_score + 3) / 6))  # Map [-3,3] to [0,1]
                else:
                    normalized = 0.5
                
                normalized_scores[metric] = normalized
        
        # Calculate combined quality score
        basic_metrics = ['brightness', 'contrast', 'sharpness', 'entropy']
        medical_metrics = ['illumination_uniformity', 'vessel_visibility', 'optic_disc_visibility']
        
        basic_score = np.mean([normalized_scores.get(m, 0.5) for m in basic_metrics])
        medical_score = np.mean([normalized_scores.get(m, 0.5) for m in medical_metrics])
        
        overall_score = 0.3 * basic_score + 0.7 * medical_score
        
        # Get threshold for this DR severity
        threshold = profile['adaptive_thresholds'].get(dr_severity, 0.3)
        
        # Determine removal reasons
        removal_reasons = []
        
        # Check for severe quality issues
        if characteristics['sharpness'] < char_stats['sharpness']['percentiles']['5']:
            removal_reasons.append('extremely_blurry')
        
        if characteristics['brightness'] < 20 or characteristics['brightness'] > 240:
            removal_reasons.append('extreme_brightness')
        
        if characteristics['illumination_uniformity'] < 0.1:
            removal_reasons.append('poor_illumination')
        
        if characteristics['vessel_visibility'] < char_stats['vessel_visibility']['percentiles']['5']:
            removal_reasons.append('poor_vessel_visibility')
        
        if characteristics['extreme_brightness_pixels'] > 0.3:
            removal_reasons.append('too_many_extreme_pixels')
        
        if characteristics['file_size_mb'] < 0.1:
            removal_reasons.append('file_too_small')
        
        # Resolution check
        w, h = characteristics['resolution']
        if w < 224 or h < 224:
            removal_reasons.append('resolution_too_low')
        
        # Make recommendation
        if len(removal_reasons) >= 2:  # Multiple severe issues
            action = 'REMOVE'
            confidence = 'HIGH'
        elif overall_score < threshold:
            action = 'REMOVE'
            confidence = 'MEDIUM'
        else:
            action = 'KEEP'
            confidence = 'HIGH' if overall_score > threshold + 0.1 else 'MEDIUM'
        
        return {
            'overall_score': overall_score,
            'threshold': threshold,
            'action': action,
            'reasons': removal_reasons,
            'confidence': confidence,
            'normalized_scores': normalized_scores
        }
    
    def create_flagged_samples(self, results, n_samples_per_dataset=20):
        """Create visual samples of flagged images for review"""
        logger.info("Creating flagged image samples for review...")
        
        # Group by dataset
        by_dataset = {}
        for result in results:
            dataset = result['dataset']
            if dataset not in by_dataset:
                by_dataset[dataset] = []
            by_dataset[dataset].append(result)
        
        for dataset_name, dataset_results in by_dataset.items():
            # Get flagged images
            flagged = [r for r in dataset_results if r['recommended_action'] == 'REMOVE']
            
            if not flagged:
                continue
            
            # Sample different types of issues
            sample_dir = f'{self.output_dir}/flagged_samples/{dataset_name}'
            os.makedirs(sample_dir, exist_ok=True)
            
            # Group by removal reasons
            by_reason = {}
            for result in flagged:
                for reason in result['removal_reasons']:
                    if reason not in by_reason:
                        by_reason[reason] = []
                    by_reason[reason].append(result)
            
            # Sample from each reason category
            samples_copied = 0
            for reason, reason_results in by_reason.items():
                reason_samples = min(5, len(reason_results), n_samples_per_dataset - samples_copied)
                if reason_samples <= 0:
                    continue
                
                # Sort by confidence and take most confident removals
                reason_results.sort(key=lambda x: x['overall_quality_score'])
                
                for i, result in enumerate(reason_results[:reason_samples]):
                    try:
                        src_path = result['image_path']
                        dst_filename = f'{reason}_{i:02d}_{result["filename"]}'
                        dst_path = os.path.join(sample_dir, dst_filename)
                        shutil.copy2(src_path, dst_path)
                        samples_copied += 1
                    except Exception as e:
                        logger.error(f"Error copying flagged sample {src_path}: {e}")
            
            logger.info(f"  Created {samples_copied} flagged samples for {dataset_name}")
    
    def generate_identification_report(self, all_results):
        """Generate comprehensive identification report"""
        logger.info("Generating identification report...")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_results)
        
        # Overall statistics
        total_images = len(df)
        flagged_for_removal = len(df[df['recommended_action'] == 'REMOVE'])
        removal_rate = flagged_for_removal / total_images if total_images > 0 else 0
        
        report = {
            'analysis_summary': {
                'total_images_analyzed': total_images,
                'images_flagged_for_removal': flagged_for_removal,
                'overall_removal_rate': removal_rate,
                'analysis_date': datetime.now().isoformat()
            },
            'dataset_breakdown': {},
            'removal_reasons_summary': {},
            'quality_score_statistics': {},
            'recommendations': []
        }
        
        # Per-dataset breakdown
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            dataset_flagged = len(dataset_data[dataset_data['recommended_action'] == 'REMOVE'])
            dataset_total = len(dataset_data)
            
            # Per-class breakdown
            class_breakdown = {}
            for dr_class in range(5):
                class_data = dataset_data[dataset_data['dr_severity'] == dr_class]
                if len(class_data) > 0:
                    class_flagged = len(class_data[class_data['recommended_action'] == 'REMOVE'])
                    class_breakdown[dr_class] = {
                        'total': len(class_data),
                        'flagged': class_flagged,
                        'removal_rate': class_flagged / len(class_data)
                    }
            
            report['dataset_breakdown'][dataset] = {
                'total_images': dataset_total,
                'flagged_images': dataset_flagged,
                'removal_rate': dataset_flagged / dataset_total if dataset_total > 0 else 0,
                'class_breakdown': class_breakdown,
                'avg_quality_score': float(dataset_data['overall_quality_score'].mean()),
                'quality_score_std': float(dataset_data['overall_quality_score'].std())
            }
        
        # Removal reasons summary
        all_reasons = []
        for _, row in df.iterrows():
            if row['recommended_action'] == 'REMOVE':
                all_reasons.extend(row['removal_reasons'])
        
        reason_counts = Counter(all_reasons)
        report['removal_reasons_summary'] = dict(reason_counts)
        
        # Quality score statistics
        report['quality_score_statistics'] = {
            'mean': float(df['overall_quality_score'].mean()),
            'std': float(df['overall_quality_score'].std()),
            'min': float(df['overall_quality_score'].min()),
            'max': float(df['overall_quality_score'].max()),
            'percentiles': {
                '10': float(df['overall_quality_score'].quantile(0.1)),
                '25': float(df['overall_quality_score'].quantile(0.25)),
                '50': float(df['overall_quality_score'].quantile(0.5)),
                '75': float(df['overall_quality_score'].quantile(0.75)),
                '90': float(df['overall_quality_score'].quantile(0.9))
            }
        }
        
        # Generate recommendations
        if removal_rate > 0.4:
            report['recommendations'].append("HIGH removal rate detected. Consider relaxing quality thresholds.")
        
        if removal_rate < 0.05:
            report['recommendations'].append("LOW removal rate detected. Consider tightening quality thresholds.")
        
        for dataset, stats in report['dataset_breakdown'].items():
            if stats['removal_rate'] > 0.5:
                report['recommendations'].append(f"Very high removal rate for {dataset}. Review dataset-specific thresholds.")
            
            # Check for class imbalance in removal
            class_rates = [info['removal_rate'] for info in stats['class_breakdown'].values()]
            if class_rates and max(class_rates) - min(class_rates) > 0.3:
                report['recommendations'].append(f"Uneven removal rates across DR classes in {dataset}. Consider class-specific adjustments.")
        
        return report
    
    def save_identification_results(self, results, report):
        """Save all identification results"""
        
        # Save detailed results
        df = pd.DataFrame(results)
        df.to_csv(f'{self.output_dir}/quality_identification_results.csv', index=False)
        
        # Save dataset profiles
        profiles_file = f'{self.output_dir}/dataset_profiles.json'
        with open(profiles_file, 'w') as f:
            json.dump(self.dataset_profiles, f, indent=2)
        
        # Save identification report
        report_file = f'{self.output_dir}/identification_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create summary CSV for easy review
        summary_data = []
        for _, row in df.iterrows():
            if row['recommended_action'] == 'REMOVE':
                summary_data.append({
                    'dataset': row['dataset'],
                    'filename': row['filename'],
                    'dr_severity': row['dr_severity'],
                    'quality_score': row['overall_quality_score'],
                    'confidence': row['confidence'],
                    'reasons': ', '.join(row['removal_reasons']),
                    'image_path': row['image_path']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{self.output_dir}/flagged_images_summary.csv', index=False)
        
        logger.info(f"\nIdentification results saved to:")
        logger.info(f"  - Detailed results: {self.output_dir}/quality_identification_results.csv")
        logger.info(f"  - Flagged summary: {self.output_dir}/flagged_images_summary.csv")
        logger.info(f"  - Dataset profiles: {self.output_dir}/dataset_profiles.json")
        logger.info(f"  - Analysis report: {self.output_dir}/identification_report.json")
        logger.info(f"  - Sample images: {self.output_dir}/sample_images/")
        logger.info(f"  - Flagged samples: {self.output_dir}/flagged_samples/")
    
    def run_identification(self, datasets_config):
        """Run complete identification process"""
        logger.info("=== STAGE 1: QUALITY ISSUE IDENTIFICATION ===")
        logger.info("This stage will ONLY identify issues - no images will be removed")
        
        all_results = []
        
        # Step 1: Characterize each dataset
        logger.info("\nStep 1: Characterizing datasets...")
        for dataset_name, dataset_path in datasets_config.items():
            if not os.path.exists(dataset_path):
                logger.warning(f"Path {dataset_path} not found, skipping {dataset_name}")
                continue
            
            profile = self.characterize_dataset(dataset_path, dataset_name)
            if profile:
                self.dataset_profiles[dataset_name] = profile
                logger.info(f"  ✓ {dataset_name} characterized ({profile['n_samples_analyzed']} samples)")
            else:
                logger.error(f"  ✗ Failed to characterize {dataset_name}")
        
        if not self.dataset_profiles:
            logger.error("No datasets could be characterized. Please check your paths and DR label extraction logic.")
            return None, None
        
        # Step 2: Identify quality issues
        logger.info("\nStep 2: Identifying quality issues...")
        for dataset_name, profile in self.dataset_profiles.items():
            dataset_path = datasets_config[dataset_name]
            results = self.identify_quality_issues(dataset_path, dataset_name, profile)
            all_results.extend(results)
        
        if not all_results:
            logger.error("No results generated. Please check your dataset structure and DR label extraction.")
            return None, None
        
        # Step 3: Create flagged samples
        logger.info("\nStep 3: Creating flagged image samples...")
        self.create_flagged_samples(all_results)
        
        # Step 4: Generate report
        logger.info("\nStep 4: Generating identification report...")
        report = self.generate_identification_report(all_results)
        
        # Step 5: Save everything
        logger.info("\nStep 5: Saving identification results...")
        self.save_identification_results(all_results, report)
        
        # Print summary
        logger.info(f"\n=== IDENTIFICATION COMPLETE ===")
        total_images = len(all_results)
        flagged_images = len([r for r in all_results if r['recommended_action'] == 'REMOVE'])
        
        logger.info(f"Total images analyzed: {total_images}")
        logger.info(f"Images flagged for removal: {flagged_images} ({flagged_images/total_images*100:.1f}%)")
        
        logger.info(f"\nNext steps:")
        logger.info(f"1. Review flagged samples in: {self.output_dir}/flagged_samples/")
        logger.info(f"2. Check summary in: {self.output_dir}/flagged_images_summary.csv")
        logger.info(f"3. Adjust thresholds if needed in dataset profiles")
        logger.info(f"4. Run stage2_remove_images.py when ready to proceed")
        
        return all_results, report

def main():
    """
    Main function - CUSTOMIZE THE DATASET PATHS AND CONFIGURATION
    """
    
    # Configuration - UPDATE THESE PATHS TO MATCH YOUR DATASETS
    datasets_config = {
        # Example configurations - adjust based on your actual dataset structure:
        
        # If your datasets are organized by folders named with DR severity:
        # 'APTOS2019': '/path/to/APTOS2019',
        # 'DeepDRiD': '/path/to/DeepDRiD', 
        # 'IDRiD': '/path/to/IDRiD',
        # 'Messidor2': '/path/to/Messidor2',
        # 'SUSTech_SYSU': '/path/to/SUSTech_SYSU'
        
        # For testing with a small dataset:
        'test_dataset': 'test_images'  # Replace with your actual path
    }
    
    # You can also customize these parameters:
    output_directory = 'quality_review'  # Where to save results
    random_seed = 42  # For reproducible sampling
    
    try:
        # Initialize identifier
        identifier = QualityIdentifier(
            output_dir=output_directory, 
            random_seed=random_seed
        )
        
        # Run identification process
        results, report = identifier.run_identification(datasets_config)
        
        if results:
            logger.info(f"\n🎯 SUCCESS: Quality identification complete!")
            logger.info(f"📁 Review results in: {output_directory}/")
            
            # Print key statistics
            total_images = len(results)
            flagged_images = len([r for r in results if r['recommended_action'] == 'REMOVE'])
            
            print(f"\n=== SUMMARY ===")
            print(f"Total images processed: {total_images:,}")
            print(f"Images flagged for removal: {flagged_images:,}")
            print(f"Removal percentage: {flagged_images/total_images*100:.1f}%")
            
            # Show per-dataset breakdown
            from collections import defaultdict
            dataset_stats = defaultdict(lambda: {'total': 0, 'flagged': 0})
            
            for result in results:
                dataset = result['dataset']
                dataset_stats[dataset]['total'] += 1
                if result['recommended_action'] == 'REMOVE':
                    dataset_stats[dataset]['flagged'] += 1
            
            print(f"\nPer-dataset breakdown:")
            for dataset, stats in dataset_stats.items():
                removal_rate = stats['flagged'] / stats['total'] * 100 if stats['total'] > 0 else 0
                print(f"  {dataset}: {stats['flagged']:,}/{stats['total']:,} ({removal_rate:.1f}%)")
            
        else:
            logger.error(f"\n❌ FAILED: No results generated")
            print(f"\nTroubleshooting tips:")
            print(f"1. Check that your dataset paths exist")
            print(f"2. Verify your DR label extraction logic matches your dataset structure")
            print(f"3. Ensure you have valid image files (.jpg, .png, etc.)")
            print(f"4. Check the logs above for specific error messages")
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()