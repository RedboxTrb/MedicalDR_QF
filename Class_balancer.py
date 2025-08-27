#!/usr/bin/env python3
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalDRBalancer:
    """
    Medical-safe diabetic retinopathy dataset balancer.
    
    Uses only augmentations that preserve anatomical validity for fundus images.
    """
    
    QUALITY_THRESHOLD = 0.7
    ROTATION_RANGE = (-12, 12)  # degrees
    BRIGHTNESS_FACTOR = 15
    CONTRAST_RANGE = (0.8, 1.2)
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_GRID_SIZE = (8, 8)
    
    def __init__(self, source_dir, output_dir, target_distribution=None, random_seed=42, quality_threshold=None):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.quality_threshold = quality_threshold or self.QUALITY_THRESHOLD
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        os.makedirs(output_dir, exist_ok=True)
        
        self.dr_classes = {
            0: "No_DR", 1: "Mild_DR", 2: "Moderate_DR", 
            3: "Severe_DR", 4: "Proliferative_DR"
        }
        
        self.current_distribution = self._detect_current_distribution()
        
        # Default target distribution
        self.target_distribution = target_distribution or {
            0: 900, 1: 750, 2: 800, 3: 750, 4: 800
        }
        
        self.augmentation_stats = defaultdict(dict)
        self.quality_stats = defaultdict(list)
        
        logger.info(f"Balancer initialized - Source: {source_dir}, Output: {output_dir}")
        logger.info(f"Quality threshold: {self.quality_threshold}")
    
    def _detect_current_distribution(self):
        """Detect current class distribution from source directory."""
        distribution = {}
        image_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')
        
        for dr_class in range(5):
            class_dir = os.path.join(self.source_dir, str(dr_class))
            if os.path.exists(class_dir):
                try:
                    image_count = len([f for f in os.listdir(class_dir) 
                                     if f.lower().endswith(image_extensions)])
                    distribution[dr_class] = image_count
                except OSError as e:
                    logger.error(f"Error reading directory {class_dir}: {e}")
                    distribution[dr_class] = 0
            else:
                distribution[dr_class] = 0
                
        return distribution
    
    def analyze_distribution_strategy(self):
        """Analyze current vs target distribution."""
        logger.info("Distribution Analysis:")
        logger.info("Class | Current | Target | Action")
        logger.info("-" * 35)
        
        for dr_class in sorted(self.dr_classes.keys()):
            current = self.current_distribution[dr_class]
            target = self.target_distribution[dr_class]
            diff = target - current
            
            if diff > 0:
                action = f"Augment +{diff}"
            elif diff < 0:
                action = f"Sample {target}"
            else:
                action = "Use all"
                
            logger.info(f"{dr_class:5} | {current:7} | {target:6} | {action}")
        
        total_target = sum(self.target_distribution.values())
        imbalance_ratio = max(self.target_distribution.values()) / min(self.target_distribution.values())
        logger.info(f"Target total: {total_target:,} images")
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        return sum(max(0, target - current) for target, current in 
                  zip(self.target_distribution.values(), self.current_distribution.values()))
    
    def create_medical_safe_augmentation(self):
        """Create augmentation functions that preserve anatomical validity."""
        
        def small_rotation(image, angle_range=None):
            if angle_range is None:
                angle_range = self.ROTATION_RANGE
            
            angle = np.random.uniform(angle_range[0], angle_range[1])
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
            return rotated
        
        def adjust_brightness_contrast(image, severity_factor=1.0):
            brightness_range = (-self.BRIGHTNESS_FACTOR * severity_factor, 
                              self.BRIGHTNESS_FACTOR * severity_factor)
            
            if severity_factor > 0.8:
                contrast_range = (0.85, 1.15)
            else:
                contrast_range = self.CONTRAST_RANGE
            
            brightness = np.random.uniform(brightness_range[0], brightness_range[1])
            contrast = np.random.uniform(contrast_range[0], contrast_range[1])
            
            return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        
        def enhance_green_channel(image):
            """CLAHE enhancement on green channel."""
            if len(image.shape) == 3:
                b, g, r = cv2.split(image)
                clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP_LIMIT, 
                                      tileGridSize=self.CLAHE_GRID_SIZE)
                g_enhanced = clahe.apply(g)
                return cv2.merge([b, g_enhanced, r])
            return image
        
        def add_minimal_noise(image, noise_level=0.3):
            noise_var = np.random.uniform(5, 12) * noise_level
            noise = np.random.normal(0, noise_var, image.shape).astype(np.float32)
            
            if len(noise.shape) == 3:
                filtered_channels = []
                for i in range(noise.shape[2]):
                    filtered_ch = cv2.bilateralFilter(noise[:,:,i].astype(np.uint8), 5, 50, 50)
                    filtered_channels.append(filtered_ch)
                filtered_noise = np.stack(filtered_channels, axis=2)
            else:
                filtered_noise = cv2.bilateralFilter(noise.astype(np.uint8), 5, 50, 50)
            
            result = cv2.add(image, filtered_noise.astype(np.uint8))
            return result
        
        def shift_color_temperature(image, shift_range=(-300, 300)):
            shift = np.random.uniform(shift_range[0], shift_range[1])
            img_float = image.astype(np.float32)
            
            if shift > 0:
                img_float[:, :, 2] = img_float[:, :, 2] * (1 + shift/1000)
                img_float[:, :, 0] = img_float[:, :, 0] * (1 - shift/2000)
            else:
                img_float[:, :, 0] = img_float[:, :, 0] * (1 + abs(shift)/1000)
                img_float[:, :, 2] = img_float[:, :, 2] * (1 - abs(shift)/2000)
            
            return np.clip(img_float, 0, 255).astype(np.uint8)
        
        def zoom_and_crop(image, zoom_range=(0.9, 1.05)):
            zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
            h, w = image.shape[:2]
            
            if abs(zoom_factor - 1.0) < 0.01:
                return image
            
            if zoom_factor < 1.0:
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                if new_h > 0 and new_w > 0:
                    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    
                    pad_y = h - new_h
                    pad_x = w - new_w
                    top_pad = pad_y // 2
                    left_pad = pad_x // 2
                    bottom_pad = pad_y - top_pad
                    right_pad = pad_x - left_pad
                    
                    return cv2.copyMakeBorder(resized, top_pad, bottom_pad, left_pad, right_pad,
                                            cv2.BORDER_REFLECT_101)
            else:
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                if new_h > h and new_w > w:
                    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    
                    start_y = (new_h - h) // 2
                    start_x = (new_w - w) // 2
                    
                    return resized[start_y:start_y + h, start_x:start_x + w]
            
            return image
        
        def gamma_adjust(image, gamma_range=(0.8, 1.2)):
            gamma = np.random.uniform(gamma_range[0], gamma_range[1])
            
            if abs(gamma - 1.0) < 0.01:
                return image
            
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        
        aug_functions = {
            'rotation': small_rotation,
            'brightness_contrast': adjust_brightness_contrast,
            'green_enhance': enhance_green_channel,
            'add_noise': add_minimal_noise,
            'color_temp': shift_color_temperature,
            'zoom': zoom_and_crop,
            'gamma': gamma_adjust
        }
        
        # Severity-specific combinations
        severity_combos = {
            0: [['brightness_contrast'], ['green_enhance'], ['rotation', 'brightness_contrast'],
                ['zoom', 'color_temp'], ['add_noise', 'gamma'], ['color_temp'], ['gamma']],
            1: [['rotation'], ['brightness_contrast'], ['green_enhance'], ['zoom'],
                ['rotation', 'brightness_contrast'], ['color_temp'], ['gamma']],
            2: [['rotation'], ['brightness_contrast'], ['green_enhance'], 
                ['zoom'], ['gamma'], ['color_temp']],
            3: [['rotation'], ['brightness_contrast'], ['green_enhance'], 
                ['zoom'], ['gamma']],
            4: [['rotation'], ['brightness_contrast'], ['green_enhance'], ['gamma']]
        }
        
        return aug_functions, severity_combos
    
    def check_augmentation_quality(self, original, augmented):
        """Check if augmented image maintains sufficient quality."""
        try:
            if len(original.shape) == 3:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original
                
            if len(augmented.shape) == 3:
                aug_gray = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
            else:
                aug_gray = augmented
            
            # Sharpness preservation
            orig_sharp = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
            aug_sharp = cv2.Laplacian(aug_gray, cv2.CV_64F).var()
            sharp_ratio = aug_sharp / max(orig_sharp, 1e-6)
            
            # Brightness preservation
            orig_bright = np.mean(orig_gray)
            aug_bright = np.mean(aug_gray)
            bright_diff = abs(orig_bright - aug_bright) / 255.0
            
            # Contrast preservation
            orig_contrast = np.std(orig_gray)
            aug_contrast = np.std(aug_gray)
            contrast_ratio = aug_contrast / max(orig_contrast, 1e-6)
            
            quality_score = (
                min(sharp_ratio, 1.0) * 0.4 + 
                (1 - bright_diff) * 0.3 + 
                min(contrast_ratio, 1.2) * 0.3
            )
            
            is_good = quality_score > self.quality_threshold
            return is_good, quality_score
            
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return False, 0.0
    
    def process_dr_class(self, dr_class, target_count):
        """Process one DR class with quality control."""
        current_count = self.current_distribution[dr_class]
        class_name = self.dr_classes[dr_class]
        
        logger.info(f"Processing Class {dr_class} ({class_name}): {current_count:,} -> {target_count:,}")
        
        source_dir = os.path.join(self.source_dir, str(dr_class))
        output_dir = os.path.join(self.output_dir, str(dr_class))
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(source_dir):
            logger.error(f"Source directory missing: {source_dir}")
            return 0
        
        img_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')
        try:
            all_images = [f for f in os.listdir(source_dir) 
                         if f.lower().endswith(img_extensions)]
        except OSError as e:
            logger.error(f"Error reading source directory {source_dir}: {e}")
            return 0
        
        if not all_images:
            logger.error(f"No images found in {source_dir}")
            return 0
        
        random.shuffle(all_images)
        created_count = self.create_balanced_subset(dr_class, all_images, target_count, output_dir)
        
        logger.info(f"Completed: {created_count:,} images created")
        return created_count
    
    def create_balanced_subset(self, dr_class, source_images, target_count, output_dir):
        """Create balanced subset with augmentation."""
        current_count = len(source_images)
        source_dir = os.path.join(self.source_dir, str(dr_class))
        
        aug_functions, severity_combos = self.create_medical_safe_augmentation()
        created_count = 0
        quality_scores = []
        
        if current_count >= target_count:
            selected_images = random.sample(source_images, target_count)
            for img_file in selected_images:
                try:
                    src_path = os.path.join(source_dir, img_file)
                    dst_path = os.path.join(output_dir, f"orig_{img_file}")
                    shutil.copy2(src_path, dst_path)
                    created_count += 1
                except (OSError, IOError) as e:
                    logger.error(f"Copy failed {img_file}: {e}")
        else:
            # Copy originals first
            originals_copied = 0
            for img_file in source_images:
                if originals_copied < target_count:
                    try:
                        src_path = os.path.join(source_dir, img_file)
                        dst_path = os.path.join(output_dir, f"orig_{img_file}")
                        shutil.copy2(src_path, dst_path)
                        originals_copied += 1
                        created_count += 1
                    except (OSError, IOError) as e:
                        logger.error(f"Copy failed {img_file}: {e}")
            
            # Generate augmented images
            need_augmented = target_count - originals_copied
            augmented_created = 0
            max_attempts = need_augmented * 3
            attempts = 0
            fails_in_a_row = 0
            
            logger.info(f"Generating {need_augmented:,} augmented images...")
            
            while augmented_created < need_augmented and attempts < max_attempts and fails_in_a_row < 50:
                attempts += 1
                
                source_img_file = random.choice(source_images)
                source_img_path = os.path.join(source_dir, source_img_file)
                
                try:
                    orig_img = cv2.imread(source_img_path)
                    if orig_img is None:
                        fails_in_a_row += 1
                        continue
                    
                    fails_in_a_row = 0
                    
                    available_combos = severity_combos[dr_class]
                    combo = random.choice(available_combos)
                    aug_img = orig_img.copy()
                    combo_name = "_".join(combo)
                    severity_factor = (dr_class + 1) / 5.0
                    
                    try:
                        for aug_name in combo:
                            if aug_name == 'brightness_contrast':
                                aug_img = aug_functions[aug_name](aug_img, severity_factor)
                            elif aug_name in aug_functions:
                                aug_img = aug_functions[aug_name](aug_img)
                        
                        quality_ok, quality_score = self.check_augmentation_quality(orig_img, aug_img)
                        
                        if quality_ok:
                            base_name = os.path.splitext(source_img_file)[0]
                            ext = os.path.splitext(source_img_file)[1]
                            aug_filename = f"aug_{combo_name}_{augmented_created:04d}_{base_name}{ext}"
                            aug_path = os.path.join(output_dir, aug_filename)
                            
                            if cv2.imwrite(aug_path, aug_img):
                                augmented_created += 1
                                created_count += 1
                                quality_scores.append(quality_score)
                                
                                if augmented_created % 100 == 0:
                                    progress = (augmented_created / need_augmented) * 100
                                    logger.info(f"Progress: {augmented_created}/{need_augmented} ({progress:.1f}%)")
                            else:
                                logger.error(f"Failed to write: {aug_path}")
                    
                    except Exception as aug_error:
                        logger.error(f"Augmentation error: {aug_error}")
                        continue
                
                except Exception as e:
                    logger.error(f"Error with {source_img_file}: {e}")
                    fails_in_a_row += 1
                    continue
            
            if fails_in_a_row >= 50:
                logger.error(f"Too many failures for class {dr_class}")
        
        if quality_scores:
            self.quality_stats[dr_class].extend(quality_scores)
        
        self.augmentation_stats[dr_class] = {
            'target_count': target_count,
            'total_created': created_count,
            'original_count': min(current_count, target_count),
            'augmented_count': max(0, created_count - min(current_count, target_count))
        }
        
        return created_count
    
    def run_balancing(self):
        """Execute complete balancing process."""
        logger.info("Medical-Safe DR Dataset Balancing")
        logger.info("-" * 40)
        
        start_time = time.time()
        self.analyze_distribution_strategy()
        
        logger.info("Processing with anatomically-safe augmentations")
        
        total_images = 0
        for dr_class in sorted(self.dr_classes.keys()):
            target_count = self.target_distribution[dr_class]
            images_created = self.process_dr_class(dr_class, target_count)
            total_images += images_created
        
        processing_time = time.time() - start_time
        self.generate_report(total_images, processing_time)
        
        return total_images, processing_time
    
    def generate_report(self, total_images, processing_time):
        """Generate final report."""
        logger.info("Balancing Complete")
        logger.info(f"Processing time: {processing_time/60:.1f} minutes")
        logger.info(f"Output: {self.output_dir}")
        
        logger.info("Final Statistics:")
        for dr_class in sorted(self.dr_classes.keys()):
            class_name = self.dr_classes[dr_class]
            stats = self.augmentation_stats[dr_class]
            
            logger.info(f"Class {dr_class} ({class_name}): {stats['total_created']:,} total")
            logger.info(f"  Original: {stats['original_count']:,}, Augmented: {stats['augmented_count']:,}")
            
            if dr_class in self.quality_stats and self.quality_stats[dr_class]:
                avg_quality = np.mean(self.quality_stats[dr_class])
                min_quality = np.min(self.quality_stats[dr_class])
                logger.info(f"  Quality - Avg: {avg_quality:.3f}, Min: {min_quality:.3f}")
        
        logger.info("Summary:")
        logger.info(f"Total images: {total_images:,}")
        
        if self.augmentation_stats:
            class_counts = [stats['total_created'] for stats in self.augmentation_stats.values()]
            if min(class_counts) > 0:
                balance_ratio = max(class_counts) / min(class_counts)
                logger.info(f"Imbalance ratio: {balance_ratio:.2f}:1")
            
            total_orig = sum(stats['original_count'] for stats in self.augmentation_stats.values())
            total_aug = sum(stats['augmented_count'] for stats in self.augmentation_stats.values())
            logger.info(f"Original: {total_orig:,}, Augmented: {total_aug:,}")
            
            if total_orig > 0:
                expansion = ((total_images / total_orig) - 1) * 100
                logger.info(f"Dataset expanded by: {expansion:.1f}%")
        
        self._save_stats(total_images, processing_time)
    
    def _save_stats(self, total_images, processing_time):
        """Save comprehensive statistics."""
        stats_file = os.path.join(self.output_dir, 'balancing_stats.json')
        
        quality_summary = {}
        for dr_class, scores in self.quality_stats.items():
            if scores:
                quality_summary[dr_class] = {
                    'count': len(scores),
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores))
                }
        
        stats_data = {
            'metadata': {
                'date': datetime.now().isoformat(),
                'quality_threshold': self.quality_threshold,
                'random_seed': self.random_seed
            },
            'processing': {
                'time_minutes': processing_time / 60,
                'total_images': total_images,
                'images_per_minute': total_images / (processing_time / 60) if processing_time > 0 else 0
            },
            'distributions': {
                'original': self.current_distribution,
                'target': self.target_distribution,
                'final': {k: v['total_created'] for k, v in self.augmentation_stats.items()}
            },
            'augmentation_details': dict(self.augmentation_stats),
            'quality_scores': quality_summary,
            'paths': {
                'source': self.source_dir,
                'output': self.output_dir
            }
        }
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
            logger.info(f"Statistics saved: {stats_file}")
        except IOError as e:
            logger.error(f"Error saving statistics: {e}")

def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded: {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing configuration: {e}")
        return {}
    except Exception as e:
        logger.error(f"Configuration loading error: {e}")
        return {}

def create_example_config():
    """Create example configuration file."""
    config = {
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
    
    config_file = 'medical_balancer_config.json'
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Example configuration created: {config_file}")
        logger.info("Edit paths and settings as needed")
    except IOError as e:
        logger.error(f"Error creating configuration: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Medical-Safe DR Dataset Balancer')
    parser.add_argument('--source', type=str, help='Source dataset directory')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--config', type=str, default='medical_balancer_config.json',
                       help='Configuration file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quality-threshold', type=float, help='Quality threshold (0.5-0.9)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create example configuration and exit')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_example_config()
        return 0
    
    config = load_config(args.config)
    
    source_dir = args.source or config.get('source_directory')
    output_dir = args.output or config.get('output_directory')
    target_dist = config.get('target_distribution')
    quality_thresh = args.quality_threshold or config.get('quality_threshold')
    seed = args.seed if args.seed != 42 else config.get('random_seed', 42)
    
    if not source_dir:
        logger.error("Source directory required - use --source or config file")
        logger.info("Run with --create-config for example configuration")
        return 1
    
    if not output_dir:
        logger.error("Output directory required - use --output or config file") 
        return 1
    
    if not os.path.exists(source_dir):
        logger.error(f"Source directory not found: {source_dir}")
        return 1
    
    if target_dist:
        try:
            target_dist = {int(k): v for k, v in target_dist.items() if k.isdigit()}
        except (ValueError, AttributeError):
            logger.error("Invalid target distribution format")
            target_dist = None
    
    if quality_thresh and not 0.5 <= quality_thresh <= 0.9:
        logger.warning(f"Quality threshold {quality_thresh} outside recommended range 0.5-0.9")
    
    logger.info("Medical-Safe DR Dataset Balancer")
    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")
    
    try:
        balancer = MedicalDRBalancer(
            source_dir=source_dir,
            output_dir=output_dir,
            target_distribution=target_dist,
            random_seed=seed,
            quality_threshold=quality_thresh
        )
        
        total_images, proc_time = balancer.run_balancing()
        logger.info("Balancing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Balancing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
