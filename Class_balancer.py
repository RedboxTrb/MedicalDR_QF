#!/usr/bin/env python3
"""
Medical-Safe DR Dataset Balancer - Standalone Version

Professional dataset balancing module for diabetic retinopathy fundus images
using only anatomically-preserving augmentations suitable for medical imaging.

Author: Research Team
Date: 2024
"""

import os
import cv2
import numpy as np
import random
import time
import shutil
import argparse
import json
from datetime import datetime
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalDRBalancer:
    """
    Medical-safe diabetic retinopathy dataset balancer that applies only
    anatomically-preserving augmentations suitable for fundus images.
    """
    
    def __init__(self, source_dir, output_dir, target_distribution=None, random_seed=42):
        """
        Initialize the medical DR balancer.
        
        Args:
            source_dir: Source directory containing DR classes (0-4 folders)
            output_dir: Output directory for balanced dataset
            target_distribution: Custom target distribution dict (optional)
            random_seed: Random seed for reproducibility
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.random_seed = random_seed
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        os.makedirs(output_dir, exist_ok=True)
        
        self.dr_classes = {
            0: "No_DR", 1: "Mild_DR", 2: "Moderate_DR", 
            3: "Severe_DR", 4: "Proliferative_DR"
        }
        
        # Auto-detect current distribution from source directory
        self.current_distribution = self._detect_current_distribution()
        
        # Default target distribution optimized for medical training
        self.target_distribution = target_distribution or {
            0: 900, 1: 750, 2: 800, 3: 750, 4: 800
        }
        
        self.augmentation_stats = defaultdict(dict)
        self.quality_stats = defaultdict(list)
        
        logger.info(f"Medical DR Balancer initialized - Source: {source_dir}, Output: {output_dir}")
    
    def _detect_current_distribution(self):
        """Automatically detect current class distribution from source directory."""
        distribution = {}
        image_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')
        
        for dr_class in range(5):
            class_dir = os.path.join(self.source_dir, str(dr_class))
            if os.path.exists(class_dir):
                image_count = len([f for f in os.listdir(class_dir) 
                                 if f.lower().endswith(image_extensions)])
                distribution[dr_class] = image_count
            else:
                distribution[dr_class] = 0
                
        return distribution
    
    def analyze_distribution_strategy(self):
        """Analyze current vs target distribution and calculate augmentation requirements."""
        print("Distribution Analysis:")
        print("Class | Current | Target | Action")
        print("-" * 35)
        
        for dr_class in sorted(self.dr_classes.keys()):
            current = self.current_distribution[dr_class]
            target = self.target_distribution[dr_class]
            difference = target - current
            
            if difference > 0:
                action = f"Augment +{difference}"
            elif difference < 0:
                action = f"Sample {target}"
            else:
                action = "Use all"
                
            print(f"{dr_class:5} | {current:7} | {target:6} | {action}")
        
        total_target = sum(self.target_distribution.values())
        imbalance_ratio = max(self.target_distribution.values()) / min(self.target_distribution.values())
        print(f"\nTarget total: {total_target:,} images")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        return sum(max(0, target - current) for target, current in 
                  zip(self.target_distribution.values(), self.current_distribution.values()))
    
    def create_medical_safe_augmentation(self):
        """
        Create augmentation pipeline with only medical-appropriate transformations.
        Excludes flips and elastic deformations that could create anatomically impossible configurations.
        """
        
        def quality_preserving_rotation(image, angle_range=(-12, 12)):
            """Apply small rotation suitable for fundus images."""
            angle = np.random.uniform(angle_range[0], angle_range[1])
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
            return rotated
        
        def medical_brightness_contrast(image, severity_factor=1.0):
            """Apply conservative brightness/contrast adjustment based on DR severity."""
            brightness_range = (-15 * severity_factor, 15 * severity_factor)
            contrast_range = (0.85, 1.15) if severity_factor > 0.8 else (0.8, 1.2)
            
            brightness = np.random.uniform(brightness_range[0], brightness_range[1])
            contrast = np.random.uniform(contrast_range[0], contrast_range[1])
            
            return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        
        def fundus_specific_enhancement(image):
            """Apply CLAHE enhancement to green channel (standard in ophthalmology)."""
            if len(image.shape) == 3:
                b, g, r = cv2.split(image)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                g_enhanced = clahe.apply(g)
                return cv2.merge([b, g_enhanced, r])
            return image
        
        def vessel_preserving_noise(image, noise_level=0.3):
            """Add minimal noise while preserving vessel structures."""
            noise_variance = np.random.uniform(5, 12) * noise_level
            noise = np.random.normal(0, noise_variance, image.shape).astype(np.float32)
            filtered_noise = cv2.bilateralFilter(noise.astype(np.uint8), 5, 50, 50)
            return cv2.add(image, filtered_noise.astype(np.uint8))
        
        def color_temperature_shift(image, shift_range=(-300, 300)):
            """Simulate different fundus camera color temperature conditions."""
            shift = np.random.uniform(shift_range[0], shift_range[1])
            img_float = image.astype(np.float32)
            
            if shift > 0:  # Warmer
                img_float[:, :, 2] *= (1 + shift/1000)  # Increase red
                img_float[:, :, 0] *= (1 - shift/2000)  # Decrease blue
            else:  # Cooler
                img_float[:, :, 0] *= (1 + abs(shift)/1000)  # Increase blue
                img_float[:, :, 2] *= (1 - abs(shift)/2000)  # Decrease red
            
            return np.clip(img_float, 0, 255).astype(np.uint8)
        
        def zoom_with_crop_variation(image, zoom_range=(0.9, 1.05)):
            """Apply medical-appropriate zoom variations."""
            zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
            height, width = image.shape[:2]
            
            if zoom_factor < 1.0:  # Zoom out
                new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                pad_y = height - new_height
                pad_x = width - new_width
                top_pad = np.random.randint(0, pad_y + 1)
                left_pad = np.random.randint(0, pad_x + 1)
                bottom_pad = pad_y - top_pad
                right_pad = pad_x - left_pad
                
                return cv2.copyMakeBorder(resized, top_pad, bottom_pad, left_pad, right_pad,
                                        cv2.BORDER_REFLECT_101)
            else:  # Zoom in
                new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                max_y = new_height - height
                max_x = new_width - width
                start_y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
                start_x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
                
                return resized[start_y:start_y + height, start_x:start_x + width]
        
        augmentation_functions = {
            'basic_rotation': quality_preserving_rotation,
            'brightness_contrast': medical_brightness_contrast,
            'fundus_enhancement': fundus_specific_enhancement,
            'vessel_noise': vessel_preserving_noise,
            'color_temperature': color_temperature_shift,
            'zoom_crop': zoom_with_crop_variation,
            'gamma_correction': lambda img: self.apply_gamma_correction(img, (0.8, 1.2))
        }
        
        # Severity-specific augmentation combinations
        severity_combos = {
            0: [['brightness_contrast'], ['fundus_enhancement'], ['basic_rotation', 'brightness_contrast'],
                ['zoom_crop', 'color_temperature'], ['vessel_noise', 'gamma_correction'], 
                ['color_temperature'], ['gamma_correction']],
            1: [['basic_rotation'], ['brightness_contrast'], ['fundus_enhancement'], ['zoom_crop'],
                ['basic_rotation', 'brightness_contrast'], ['color_temperature'], ['gamma_correction']],
            2: [['basic_rotation'], ['brightness_contrast'], ['fundus_enhancement'], 
                ['zoom_crop'], ['gamma_correction'], ['color_temperature']],
            3: [['basic_rotation'], ['brightness_contrast'], ['fundus_enhancement'], 
                ['zoom_crop'], ['gamma_correction']],
            4: [['basic_rotation'], ['brightness_contrast'], ['fundus_enhancement'], ['gamma_correction']]
        }
        
        return augmentation_functions, severity_combos
    
    def apply_gamma_correction(self, image, gamma_range):
        """Apply gamma correction within specified range."""
        gamma = np.random.uniform(gamma_range[0], gamma_range[1])
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def assess_augmentation_quality(self, original, augmented):
        """
        Assess quality of augmented image based on sharpness, brightness, and contrast preservation.
        Returns (is_acceptable, quality_score).
        """
        try:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
            aug_gray = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY) if len(augmented.shape) == 3 else augmented
            
            # Sharpness preservation
            orig_sharpness = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
            aug_sharpness = cv2.Laplacian(aug_gray, cv2.CV_64F).var()
            sharpness_ratio = aug_sharpness / max(orig_sharpness, 1)
            
            # Brightness preservation
            orig_brightness = np.mean(orig_gray)
            aug_brightness = np.mean(aug_gray)
            brightness_diff = abs(orig_brightness - aug_brightness) / 255.0
            
            # Contrast preservation
            orig_contrast = np.std(orig_gray)
            aug_contrast = np.std(aug_gray)
            contrast_ratio = aug_contrast / max(orig_contrast, 1)
            
            quality_score = (
                min(sharpness_ratio, 1.0) * 0.4 +  
                (1 - brightness_diff) * 0.3 +      
                min(contrast_ratio, 1.2) * 0.3     
            )
            
            return quality_score > 0.7, quality_score
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return False, 0.0
    
    def process_class_with_quality_control(self, dr_class, target_count):
        """Process DR class with quality-controlled augmentation."""
        current_count = self.current_distribution[dr_class]
        class_name = self.dr_classes[dr_class]
        
        print(f"Processing Class {dr_class} ({class_name}): {current_count:,} -> {target_count:,}")
        
        source_class_dir = os.path.join(self.source_dir, str(dr_class))
        output_class_dir = os.path.join(self.output_dir, str(dr_class))
        os.makedirs(output_class_dir, exist_ok=True)
        
        if not os.path.exists(source_class_dir):
            logger.error(f"Source directory not found: {source_class_dir}")
            return 0
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')
        all_images = [f for f in os.listdir(source_class_dir) 
                     if f.lower().endswith(image_extensions)]
        
        if not all_images:
            logger.error(f"No images found in {source_class_dir}")
            return 0
        
        random.shuffle(all_images)
        created_count = self.create_balanced_subset(dr_class, all_images, target_count, output_class_dir)
        
        print(f"Completed: {created_count:,} images created")
        return created_count
    
    def create_balanced_subset(self, dr_class, source_images, target_count, output_dir):
        """Create balanced subset with medical-safe augmentation."""
        current_count = len(source_images)
        source_class_dir = os.path.join(self.source_dir, str(dr_class))
        
        augmentation_functions, severity_combos = self.create_medical_safe_augmentation()
        created_count = 0
        quality_scores = []
        
        if current_count >= target_count:
            # Sample existing images
            selected_images = random.sample(source_images, target_count)
            for img_file in selected_images:
                try:
                    src_path = os.path.join(source_class_dir, img_file)
                    dst_path = os.path.join(output_dir, f"original_{img_file}")
                    shutil.copy2(src_path, dst_path)
                    created_count += 1
                except Exception as e:
                    logger.error(f"Error copying {img_file}: {e}")
        else:
            # Copy originals first
            originals_copied = 0
            for img_file in source_images:
                if originals_copied < target_count:
                    try:
                        src_path = os.path.join(source_class_dir, img_file)
                        dst_path = os.path.join(output_dir, f"original_{img_file}")
                        shutil.copy2(src_path, dst_path)
                        originals_copied += 1
                        created_count += 1
                    except Exception as e:
                        logger.error(f"Error copying {img_file}: {e}")
            
            # Generate augmented images
            augmented_needed = target_count - originals_copied
            augmented_created = 0
            max_attempts = augmented_needed * 3
            attempts = 0
            
            print(f"  Generating {augmented_needed:,} augmented images...")
            
            while augmented_created < augmented_needed and attempts < max_attempts:
                attempts += 1
                
                source_img_file = random.choice(source_images)
                source_img_path = os.path.join(source_class_dir, source_img_file)
                
                try:
                    original_image = cv2.imread(source_img_path)
                    if original_image is None:
                        continue
                    
                    available_combos = severity_combos[dr_class]
                    combo = random.choice(available_combos)
                    augmented_image = original_image.copy()
                    combo_name = "_".join(combo)
                    severity_factor = (dr_class + 1) / 5.0
                    
                    for aug_name in combo:
                        if aug_name == 'brightness_contrast':
                            augmented_image = augmentation_functions[aug_name](augmented_image, severity_factor)
                        elif aug_name in augmentation_functions:
                            augmented_image = augmentation_functions[aug_name](augmented_image)
                    
                    quality_ok, quality_score = self.assess_augmentation_quality(original_image, augmented_image)
                    
                    if quality_ok:
                        base_name = os.path.splitext(source_img_file)[0]
                        extension = os.path.splitext(source_img_file)[1]
                        aug_filename = f"aug_{combo_name}_{augmented_created:04d}_{base_name}{extension}"
                        aug_path = os.path.join(output_dir, aug_filename)
                        
                        cv2.imwrite(aug_path, augmented_image)
                        augmented_created += 1
                        created_count += 1
                        quality_scores.append(quality_score)
                        
                        if augmented_created % 100 == 0:
                            progress = (augmented_created / augmented_needed) * 100
                            print(f"    Progress: {augmented_created}/{augmented_needed} ({progress:.1f}%)")
                
                except Exception as e:
                    logger.error(f"Error in augmentation: {e}")
                    continue
        
        if quality_scores:
            self.quality_stats[dr_class].extend(quality_scores)
        
        self.augmentation_stats[dr_class] = {
            'target_count': target_count,
            'total_created': created_count,
            'original_count': min(current_count, target_count),
            'augmented_count': max(0, target_count - min(current_count, target_count))
        }
        
        return created_count
    
    def run_medical_balancing(self):
        """Execute the complete medical-safe balancing process."""
        print("Medical-Safe DR Dataset Balancing")
        print("-" * 40)
        
        start_time = time.time()
        self.analyze_distribution_strategy()
        
        print("\nProcessing classes with anatomically-preserving augmentations only")
        
        total_images = 0
        for dr_class in sorted(self.dr_classes.keys()):
            target_count = self.target_distribution[dr_class]
            images_created = self.process_class_with_quality_control(dr_class, target_count)
            total_images += images_created
        
        processing_time = time.time() - start_time
        self.generate_medical_report(total_images, processing_time)
        
        return total_images, processing_time
    
    def generate_medical_report(self, total_images, processing_time):
        """Generate comprehensive processing report."""
        print(f"\nBalancing Complete")
        print(f"Processing time: {processing_time/60:.1f} minutes")
        print(f"Output directory: {self.output_dir}")
        
        print(f"\nFinal Dataset Statistics:")
        for dr_class in sorted(self.dr_classes.keys()):
            class_name = self.dr_classes[dr_class]
            stats = self.augmentation_stats[dr_class]
            
            print(f"Class {dr_class} ({class_name}): {stats['total_created']:,} total")
            print(f"  Original: {stats['original_count']:,}, Augmented: {stats['augmented_count']:,}")
            
            if dr_class in self.quality_stats and self.quality_stats[dr_class]:
                avg_quality = np.mean(self.quality_stats[dr_class])
                print(f"  Average quality score: {avg_quality:.3f}")
        
        print(f"\nDataset Summary:")
        print(f"Total images: {total_images:,}")
        
        class_counts = [stats['total_created'] for stats in self.augmentation_stats.values()]
        balance_ratio = max(class_counts) / min(class_counts)
        print(f"Imbalance ratio: {balance_ratio:.2f}:1")
        
        total_original = sum(stats['original_count'] for stats in self.augmentation_stats.values())
        total_augmented = sum(stats['augmented_count'] for stats in self.augmentation_stats.values())
        print(f"Original images: {total_original:,}")
        print(f"Augmented images: {total_augmented:,}")
        print(f"Dataset expansion: {((total_images / total_original) - 1) * 100:.1f}%")
        
        # Save statistics
        stats_file = os.path.join(self.output_dir, 'balancing_statistics.json')
        stats_data = {
            'processing_date': datetime.now().isoformat(),
            'processing_time_minutes': processing_time / 60,
            'total_images': total_images,
            'augmentation_stats': dict(self.augmentation_stats),
            'quality_stats': {k: {'count': len(v), 'mean': np.mean(v) if v else 0} 
                             for k, v in self.quality_stats.items()},
            'source_directory': self.source_dir,
            'output_directory': self.output_dir,
            'current_distribution': self.current_distribution,
            'target_distribution': self.target_distribution
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"Statistics saved: {stats_file}")

def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing configuration file: {e}")
        return None

def create_example_config():
    """Create example configuration file."""
    config = {
        "source_directory": "/path/to/source/dataset",
        "output_directory": "/path/to/output/dataset",
        "target_distribution": {
            "0": 900,
            "1": 750,
            "2": 800,
            "3": 750,
            "4": 800
        },
        "random_seed": 42
    }
    
    with open('medical_balancer_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Example configuration created: medical_balancer_config.json")
    print("Update the paths and target distribution according to your needs")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Medical-Safe DR Dataset Balancer')
    parser.add_argument('--source', type=str, help='Source dataset directory')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--config', type=str, default='medical_balancer_config.json',
                       help='Configuration file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--create-config', action='store_true',
                       help='Create example configuration file')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_example_config()
        return
    
    # Load configuration if available
    config = load_config(args.config) if os.path.exists(args.config) else {}
    
    # Use command line arguments or config file values
    source_dir = args.source or config.get('source_directory')
    output_dir = args.output or config.get('output_directory')
    target_distribution = config.get('target_distribution')
    random_seed = args.seed or config.get('random_seed', 42)
    
    # Validate inputs
    if not source_dir:
        print("Error: Source directory not specified")
        print("Use --source argument or create configuration file with --create-config")
        return
    
    if not output_dir:
        print("Error: Output directory not specified")
        print("Use --output argument or create configuration file with --create-config")
        return
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    # Convert string keys to integers for target distribution
    if target_distribution:
        target_distribution = {int(k): v for k, v in target_distribution.items()}
    
    print("Medical-Safe DR Dataset Balancer")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print("Applying medical-safe augmentations (no flips, no elastic transforms)")
    
    # Initialize and run balancer
    balancer = MedicalDRBalancer(
        source_dir=source_dir,
        output_dir=output_dir,
        target_distribution=target_distribution,
        random_seed=random_seed
    )
    
    total_images, processing_time = balancer.run_medical_balancing()
    print("Medical dataset balancing completed")

if __name__ == "__main__":
    main()