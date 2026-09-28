import cv2
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
import math


def max_inscribed_square(img_width, img_height, rotation_angle_deg):
    """
    Maximum side length of an axis-aligned square (in the rotated buffer) such
    that its inverse-rotation lies entirely inside the W×H original image.

    Equivalently, the largest square whose bounding box `s · (|cos θ| + |sin θ|)`
    fits inside `min(W, H)`.

    No multiplicative safety margin: integer-quantization safety is handled by
    the caller subtracting a fixed pixel margin.
    """
    angle_rad = math.radians(rotation_angle_deg)
    factor = abs(math.cos(angle_rad)) + abs(math.sin(angle_rad))
    if factor < 1e-9:
        return float(min(img_width, img_height))
    return min(img_width, img_height) / factor


def sample_offset_in_original(img_width, img_height, crop_size,
                              rotation_angle_deg, rng):
    """
    Sample a uniform crop-center offset `(dx, dy)` in the rotated buffer such
    that the corresponding crop is fully inside the W×H original image.

    Math: in the original-image frame the crop is a square rotated by `-θ`,
    centred at `(W/2 + dx_orig, H/2 + dy_orig)`. Its axis-aligned bounding
    box has side `bbox = s · (|cos θ| + |sin θ|)`, so the valid set of
    `(dx_orig, dy_orig)` is the rectangle `|dx_orig| ≤ (W - bbox)/2` and
    `|dy_orig| ≤ (H - bbox)/2`. Sampling uniformly there and forward-rotating
    gives a uniform offset over the (rotated) admissible region in the buffer.
    """
    angle_rad = math.radians(rotation_angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    bbox = crop_size * (abs(cos_a) + abs(sin_a))

    x_max = max(0.0, (img_width - bbox) / 2.0)
    y_max = max(0.0, (img_height - bbox) / 2.0)

    dx_orig = rng.uniform(-x_max, x_max) if x_max > 0 else 0.0
    dy_orig = rng.uniform(-y_max, y_max) if y_max > 0 else 0.0

    dx = dx_orig * cos_a - dy_orig * sin_a
    dy = dx_orig * sin_a + dy_orig * cos_a
    return int(round(dx)), int(round(dy))


def rotate_and_crop_square(image, angle_deg, crop_size, offset_x=0, offset_y=0):
    """
    Rotate `image` by `angle_deg` around its centre, then extract an
    axis-aligned square of side `crop_size` at `(W/2 + offset_x, H/2 + offset_y)`
    in the rotated buffer.

    The caller must pass an offset chosen by `sample_offset_in_original` so
    the crop falls inside the rotated original. `BORDER_REFLECT_101` is set as
    a defensive backstop only — in correct usage, no reflected pixels are
    sampled.
    """
    h, w = image.shape[:2]
    center_x, center_y = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

    crop_cx = int(round(center_x + offset_x))
    crop_cy = int(round(center_y + offset_y))
    half = crop_size // 2
    x1 = crop_cx - half
    y1 = crop_cy - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # Defensive clamp to buffer bounds (correct offsets never trigger this).
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        x1 = max(0, min(x1, w - crop_size))
        y1 = max(0, min(y1, h - crop_size))
        x2 = x1 + crop_size
        y2 = y1 + crop_size

    return rotated[y1:y2, x1:x2]


def generate_augmented_crop(image_path, min_area_ratio=0.4,
                            max_rotation_attempts=200, rng=None):
    """
    Generate one augmented crop with strict guarantees:

      * Square output with side in `[ceil(sqrt(min_area_ratio · W · H)),
        floor(min(W, H) / (|cos θ| + |sin θ|)) − 2]`, so the crop area ratio
        is always ≥ `min_area_ratio` (strict floor) and ≤ the geometric
        ceiling for a square inscribed in the rotated W × H rectangle.
      * Every crop pixel comes from the original image — no
        `BORDER_REFLECT_101` content is ever sampled.
      * Rotation is uniform over the admissible angles (rejection sampled);
        for non-square images, a narrow band around 45°/135°/225°/315° is
        excluded because no square of area ≥ `min_area_ratio · W · H` fits
        inside the W × H rectangle at those angles.
    """
    if rng is None:
        rng = np.random

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]
    orig_area = float(h * w)
    min_size = int(math.ceil(math.sqrt(orig_area * min_area_ratio)))

    # Geometric upper bound (no rotation): min(W, H). If the requested
    # min_area_ratio exceeds the geometric ceiling for this aspect ratio,
    # clamp so the loop terminates.
    abs_max_size = min(w, h)
    if min_size > abs_max_size - 2:
        min_size = abs_max_size - 2

    rotation_angle = None
    max_size = None
    for _ in range(max_rotation_attempts):
        candidate_angle = rng.uniform(0.0, 360.0)
        candidate_max = max_inscribed_square(w, h, candidate_angle)
        candidate_max_int = int(math.floor(candidate_max)) - 2  # 2-px safety
        if candidate_max_int >= min_size:
            rotation_angle = candidate_angle
            max_size = candidate_max_int
            break

    if rotation_angle is None:
        # Should not happen for sensible (min_area_ratio, aspect ratio).
        # Fall back to axis-aligned (no rotation) which always achieves the
        # geometric maximum H²/(W·H).
        rotation_angle = 0.0
        max_size = abs_max_size - 2

    # Uniform crop side in [min_size, max_size]; corresponding area ratio is
    # uniform in side, NOT in area — see test in `_self_test`.
    if max_size <= min_size:
        crop_size = min_size
    else:
        crop_size = int(rng.randint(min_size, max_size + 1))

    offset_x, offset_y = sample_offset_in_original(
        w, h, crop_size, rotation_angle, rng
    )

    return rotate_and_crop_square(image, rotation_angle, crop_size,
                                  offset_x, offset_y)


def _self_test():
    """Sanity test: simulate 5,000 crops on a 2592×1944 image and verify the
    area-ratio and boundary-fit guarantees hold. Run with `python tools/augment.py --self-test`."""
    print("Running self-test (5000 sims, W=2592, H=1944, min_area=0.4)...")
    W, H = 2592, 1944
    rng = np.random.RandomState(0)
    ratios = []
    for _ in range(5000):
        rotation_angle = None
        max_size = None
        min_size = int(math.ceil(math.sqrt(W * H * 0.4)))
        for _ in range(200):
            candidate_angle = rng.uniform(0, 360)
            candidate_max = int(math.floor(max_inscribed_square(W, H, candidate_angle))) - 2
            if candidate_max >= min_size:
                rotation_angle = candidate_angle
                max_size = candidate_max
                break
        if rotation_angle is None:
            rotation_angle, max_size = 0.0, min(W, H) - 2
        crop_size = rng.randint(min_size, max_size + 1)
        ratios.append(crop_size * crop_size / (W * H))
    ratios = np.array(ratios)
    print(f"  area ratio: min={ratios.min():.4f}  median={np.median(ratios):.4f}  max={ratios.max():.4f}")
    assert ratios.min() >= 0.40 - 1e-6, f"area ratio violated: {ratios.min():.4f}"
    assert ratios.max() <= 0.75 + 1e-3, f"area ratio impossibly high: {ratios.max():.4f}"
    print(f"  PASS: every crop has area ratio in [{ratios.min():.4f}, {ratios.max():.4f}] ⊂ [0.4, 0.75]")
    print(f"  fraction with ratio ≥ 0.4: {(ratios >= 0.4).mean()*100:.1f}%  (must be 100.0%)")


def augment_dataset(input_dir, output_dir, multiplier=5, min_area_ratio=0.2, seed=42, resize_to=None, only_class=None):
    """
    Create an augmented dataset with more images through random cropping and rotation.
    
    Args:
        input_dir: Input directory containing class folders
        output_dir: Output directory for augmented dataset
        multiplier: How many times to multiply the dataset (default: 5)
        min_area_ratio: Minimum crop area ratio (default: 0.2 = 20%)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    classes = os.listdir(input_dir)
    if only_class is not None:
        classes = [c for c in classes if c == only_class]
        if not classes:
            raise ValueError(f"--only_class '{only_class}' not found in {input_dir}")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    for class_name in classes:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    print(f"Creating augmented dataset with {multiplier}x more images")
    print(f"Minimum crop area: {min_area_ratio*100:.1f}% of original")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 70)
    
    total_generated = 0
    
    for class_name in classes:
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        if not os.path.exists(input_class_dir):
            print(f"Warning: Class directory not found: {input_class_dir}")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(input_class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"Warning: No images found in {input_class_dir}")
            continue
        
        # Check if output folder exists and has augmented images
        expected_augmented_count = len(image_files) * multiplier
        if os.path.exists(output_class_dir):
            existing_augmented = [f for f in os.listdir(output_class_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            num_existing_augmented = len(existing_augmented)
            
            if num_existing_augmented >= expected_augmented_count:
                print(f"\nSkipping class '{class_name}': {len(image_files)} original images")
                print(f"  Folder already exists with {num_existing_augmented} augmented images (expected: {expected_augmented_count})")
                continue
            else:
                print(f"\nOverwriting class '{class_name}': {len(image_files)} original images")
                print(f"  Folder has only {num_existing_augmented} augmented images (expected: {expected_augmented_count})")
                # Remove existing augmented images
                for aug_file in existing_augmented:
                    os.remove(os.path.join(output_class_dir, aug_file))
        
        print(f"\nProcessing class '{class_name}': {len(image_files)} original images")
        
        # Generate augmented images
        pbar = tqdm(image_files, desc=f"Class {class_name}")
        
        for img_idx, img_file in enumerate(pbar):
            img_path = os.path.join(input_class_dir, img_file)
            base_name = Path(img_file).stem
            
            # Generate multiple augmented versions of each image
            for aug_idx in range(multiplier):
                try:
                    # Generate augmented crop
                    augmented_img = generate_augmented_crop(img_path, min_area_ratio)

                    if resize_to:
                        augmented_img = cv2.resize(augmented_img, (resize_to, resize_to),
                                                   interpolation=cv2.INTER_AREA)

                    # Save augmented image
                    output_name = f"{base_name}_aug_{aug_idx:03d}.jpg"
                    output_path = os.path.join(output_class_dir, output_name)
                    cv2.imwrite(output_path, augmented_img, 
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    total_generated += 1
                    
                except Exception as e:
                    print(f"\nError processing {img_file}: {str(e)}")
                    continue
                
                pbar.set_postfix({'generated': total_generated})
        
        # Count generated images for this class
        generated_count = len([f for f in os.listdir(output_class_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  Generated {generated_count} augmented images for class '{class_name}'")
    
    print("\n" + "=" * 70)
    print(f"Dataset augmentation completed!")
    print(f"Total images generated: {total_generated}")
    print(f"Output saved to: {output_dir}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    for class_name in classes:
        output_class_dir = os.path.join(output_dir, class_name)
        if os.path.exists(output_class_dir):
            count = len([f for f in os.listdir(output_class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  Class {class_name}: {count:,} images")


def visualize_samples(input_dir, output_dir, num_samples=5):
    """
    Generate visualization comparing original and augmented images.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        num_samples: Number of samples to visualize per class
    """
    import matplotlib.pyplot as plt
    
    classes = ['a', 'b', 'c', 'd']
    
    for class_name in classes:
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        if not os.path.exists(input_class_dir) or not os.path.exists(output_class_dir):
            continue
        
        # Get sample images
        input_images = [f for f in os.listdir(input_class_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
        
        if not input_images:
            continue
        
        fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for row, img_file in enumerate(input_images):
            # Load original image
            orig_path = os.path.join(input_class_dir, img_file)
            orig_img = cv2.imread(orig_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            # Display original
            axes[row, 0].imshow(orig_img)
            axes[row, 0].set_title('Original' if row == 0 else '')
            axes[row, 0].axis('off')
            
            # Load and display augmented versions
            base_name = Path(img_file).stem
            for aug_idx in range(5):
                aug_file = f"{base_name}_aug_{aug_idx:03d}.jpg"
                aug_path = os.path.join(output_class_dir, aug_file)
                
                if os.path.exists(aug_path):
                    aug_img = cv2.imread(aug_path)
                    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                    axes[row, aug_idx + 1].imshow(aug_img)
                    axes[row, aug_idx + 1].set_title(f'Aug {aug_idx+1}' if row == 0 else '')
                axes[row, aug_idx + 1].axis('off')
        
        plt.suptitle(f'Class {class_name}: Original vs Augmented Images', fontsize=14)
        plt.tight_layout()
        
        viz_path = os.path.join(output_dir, f'visualization_class_{class_name}.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {viz_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Augment bacteria image dataset with random rotated crops'
    )
    
    parser.add_argument('--input_dir', type=str, default='sampled_frames/multi/',
                       help='Input directory with original images (default: sampled_frames)')
    parser.add_argument('--output_dir', type=str, default='augmented_data/multi',
                       help='Output directory for augmented images (default: augmented_data)')
    parser.add_argument('--multiplier', type=int, default=3,
                       help='How many augmented images to create per original (default: 3)')
    parser.add_argument('--min_area_ratio', type=float, default=0.4,
                       help='Minimum crop area ratio (default: 0.4 = 40%%)')
    parser.add_argument('--resize_to', type=int, default=1024,
                       help='Resize each crop to this square size in pixels (default: 1024)')
    parser.add_argument('--only_class', type=str, default=None,
                       help='If set, process only this one combo subfolder (for SLURM array jobs)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization of augmented samples')
    parser.add_argument('--viz_samples', type=int, default=3,
                       help='Number of samples to visualize per class (default: 3)')
    parser.add_argument('--self-test', action='store_true',
                       help='Run a 5,000-sample sanity test and exit')

    args = parser.parse_args()

    if args.self_test:
        _self_test()
        sys.exit(0)

    # Validate arguments
    if args.multiplier < 1:
        raise ValueError("Multiplier must be at least 1")
    if not (0 < args.min_area_ratio <= 1):
        raise ValueError("min_area_ratio must be between 0 and 1")
    
    # Run augmentation
    augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        multiplier=args.multiplier,
        min_area_ratio=args.min_area_ratio,
        seed=args.seed,
        resize_to=args.resize_to,
        only_class=args.only_class,
    )
    
    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_samples(args.input_dir, args.output_dir, args.viz_samples)
        print("Visualizations completed!")
