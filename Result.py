#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOTA_VisualizationResult_Stable.py (v5 - Alignment Fix)

This version adds a robust _align_volumes helper to
center-crop or center-pad the prediction/ground truth to match
the original image, fixing any misalignment artifacts.
"""
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio
import io
import os
import yaml
from scipy.ndimage import rotate, zoom
from matplotlib import cm
import tempfile
import sys 

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

class VisualizationResult:
    """
    Handles saving, processing, and visualizing 3D segmentation results.
    This class is config-driven, 100% stable, and works for all MSD tasks.
    """
    def __init__(self,
                 config_path: str,
                 original_image: np.ndarray,
                 ground_truth: np.ndarray,
                 prediction: np.ndarray,
                 nifti_header: nib.Nifti1Header,
                 output_dir="visualization_artifacts"):
        """
        Initialize with data arrays, metadata, and the config dictionary.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.output_dir = output_dir
        self.nifti_header = nifti_header


        modality = self.config['data'].get('modality', 'MRI')
        if modality == 'MRI' and original_image.ndim == 4:
            flair_index = 3 if 'Task01' in self.config['data'].get('tarfile_name', '') else 0
            self.original_image = self._to_3d(original_image, flair_index)
        else:
            self.original_image = self._to_3d(original_image)
            

        prediction_3d = self._to_3d(prediction)
        ground_truth_3d = self._to_3d(ground_truth)
        
        self.prediction = self._align_volumes(self.original_image, prediction_3d)
        self.ground_truth = self._align_volumes(self.original_image, ground_truth_3d)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

 
        self.class_names = self.config['data']['class_names']
        self.num_classes = len(self.class_names)
        

        default_hex_colors = [
            '#000000', '#00FF00', '#FFFF00', '#FF0000', 
            '#0000FF', '#800080', '#00FFFF', '#FF00FF'
        ]

        self.class_colors_hex = {}   
        matplotlib_colors_list = [] 
        
        for i, name in enumerate(self.class_names):
            hex_color = default_hex_colors[i % len(default_hex_colors)]
            
            if i == 0: # 'BackGround'
                rgba_tuple = (0.0, 0.0, 0.0, 0.0)
            else:
                rgb = tuple(int(hex_color.lstrip('#')[j:j+2], 16) / 255.0 for j in (0, 2, 4))
                rgba_tuple = (*rgb, 1.0)
            
            matplotlib_colors_list.append(rgba_tuple)
            
            if i > 0:
                self.class_colors_hex[i] = rgba_tuple

        self.matplotlib_cmap = ListedColormap(matplotlib_colors_list)
        self.matplotlib_cmap.set_under(color=(0.0, 0.0, 0.0, 0.0))
        
        print(f"VisualizationResult initialized for {self.num_classes} classes.")
        print(f"  2D CMap Colors: {matplotlib_colors_list}")
        print(f"  3D Fallback Palette: {self.class_colors_hex}")

    def _to_3d(self, array, channel_index=0):
        """Ensure array is 3D."""
        if array.ndim == 4:
            if array.shape[0] == 1:
                return array.squeeze(0)
            elif array.shape[-1] == 1:
                return array.squeeze(-1)
            else:
                return array[..., channel_index]
        elif array.ndim == 3:
            return array
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")

    def _align_volumes(self, img_vol, seg_vol):
        """
        Center-crops or center-pads the segmentation to match the image.
        This fixes any misalignment from a buggy _postprocess function.
        """
        img_shape = img_vol.shape[:3]
        seg_shape = seg_vol.shape[:3]
        
        if img_shape == seg_shape:
            return seg_vol # Already aligned
            
        print(f"WARNING: Mismatch detected. Image shape {img_shape} != Seg shape {seg_shape}.")
        print("         Applying center-crop/pad to align visualization.")
        print("         (This is a workaround. For a permanent fix, correct your Inference.py _postprocess function)")


        aligned_seg = np.zeros(img_shape, dtype=seg_vol.dtype)
        
  
        slices_img = []
        slices_seg = []
        
        for i in range(3): 
            dim_img = img_shape[i]
            dim_seg = seg_shape[i]
            
            if dim_img > dim_seg: 
                pad_before = (dim_img - dim_seg) // 2
                pad_after = dim_img - dim_seg - pad_before
                slices_img.append(slice(pad_before, dim_img - pad_after))
                slices_seg.append(slice(None))
            else: 
                crop_before = (dim_seg - dim_img) // 2
                crop_after = dim_seg - dim_img - crop_before
                slices_img.append(slice(None)) # Use full img dim
                slices_seg.append(slice(crop_before, dim_seg - crop_after))
        

        aligned_seg[tuple(slices_img)] = seg_vol[tuple(slices_seg)]
            
        return aligned_seg


    def save_as_nifti(self, filename="prediction.nii.gz"):
        """Saves the prediction as a Nifti file."""
        output_path = os.path.join(self.output_dir, filename)
        affine = self.nifti_header.get_best_affine()
        pred_to_save = self.prediction.astype(np.int16)
        img = nib.Nifti1Image(pred_to_save, affine, self.nifti_header)
        nib.save(img, output_path)
        print(f"Prediction saved to {output_path}")


    def plot_2d_montage(self, slice_index, plane='axial', ax=None):
        """
        Plots a 2D montage (Original, GT, Pred) for a given slice.
        """
        if plane == 'axial':
            orig_slice = self.original_image[:, :, slice_index]
            gt_slice = self.ground_truth[:, :, slice_index]
            pred_slice = self.prediction[:, :, slice_index]
            title = f"Axial Slice: {slice_index}"
        elif plane == 'coronal':
            orig_slice = self.original_image[:, slice_index, :]
            gt_slice = self.ground_truth[:, slice_index, :]
            pred_slice = self.prediction[:, slice_index, :]
            title = f"Coronal Slice: {slice_index}"
        elif plane == 'sagittal':
            orig_slice = self.original_image[slice_index, :, :]
            gt_slice = self.ground_truth[slice_index, :, :]
            pred_slice = self.prediction[slice_index, :, :]
            title = f"Sagittal Slice: {slice_index}"
        else:
            raise ValueError(f"Unknown plane: {plane}")

        orig_slice = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min() + 1e-6)
        orig_slice = np.rot90(orig_slice)
        gt_slice = np.rot90(gt_slice)
        pred_slice = np.rot90(pred_slice)
        
        if ax is None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(title, fontsize=16)
            return_fig = fig
        else:
            if not isinstance(ax, (list, np.ndarray)) or len(ax) != 3:
                raise ValueError("If 'ax' is provided, it must be a list/array of 3 matplotlib axes.")
            ax1, ax2, ax3 = ax
            ax1.set_title(title, fontsize=10)
            return_fig = ax[0].get_figure()

        ax1.imshow(orig_slice, cmap='gray')
        ax1.set_title("Original Scan")
        ax1.axis('off')
        
        ax2.imshow(orig_slice, cmap='gray')
        ax2.imshow(gt_slice, cmap=self.matplotlib_cmap, vmin=0.5,
                   vmax=self.num_classes - 0.5, interpolation='none')
        ax2.set_title("Ground Truth")
        ax2.axis('off')
        
        ax3.imshow(orig_slice, cmap='gray')
        ax3.imshow(pred_slice, cmap=self.matplotlib_cmap, vmin=0.5,
                   vmax=self.num_classes - 0.5, interpolation='none')
        ax3.set_title("Model Prediction")
        ax3.axis('off')
        
        if ax is None:
            plt.tight_layout()
            return return_fig


    def create_fallback_3d_gif(self, output_filename: str, num_frames: int = 90, dpi=100):
        """
        Stable fallback GIF generator (no PyVista, no kernel crashes).
        This version assumes __init__ has already aligned the volumes.
        """
        print("🚀 Starting STABLE fallback 3D GIF creation (Matplotlib)...")
        output_path = os.path.join(self.output_dir, output_filename)
        try:
  
            img = np.asarray(self.original_image, dtype=np.float32)
            seg = np.asarray(self.prediction, dtype=np.uint8)


            img_nonzero = img[img > 0]
            lo, hi = (np.percentile(img_nonzero, (2, 98)) if img_nonzero.size 
                      else (img.min(), img.max()))
            img_disp = np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0)
            brain_mask = (img_disp > 0.05).astype(np.uint8)


            projections = [
                ("Axial", np.mean(img_disp, axis=2), np.max(seg, axis=2), np.max(brain_mask, axis=2)),
                ("Coronal", np.mean(img_disp, axis=0), np.max(seg, axis=0), np.max(brain_mask, axis=0)),
                ("Sagittal", np.mean(img_disp, axis=1), np.max(seg, axis=1), np.max(brain_mask, axis=1)),
            ]

            # --- CONFIG-DRIVEN PALETTE ---
            palette = self.class_colors_hex
            print(f"Using config-driven palette: {palette}")

            def overlay(img_gray, seg_mask, valid_mask, alpha=0.55):
                """Blend segmentation only inside valid brain region."""
                rgb = np.stack([img_gray]*3, axis=-1)
                overlay_rgb = rgb.copy()
                for lbl, col in palette.items():
                    mask = (seg_mask == lbl) & (valid_mask > 0)
                    if np.any(mask):
                        color = np.array(col[:3])
                        overlay_rgb[mask] = (1 - alpha) * overlay_rgb[mask] + alpha * color
                return np.clip(overlay_rgb, 0.0, 1.0)

            # --- Generate frames (from your code) ---
            with tempfile.TemporaryDirectory() as tmpdir:
                frames = []
                fixed_size = None
                print(f"Rendering {num_frames} frames for fallback GIF...")
                
                for i in range(num_frames):
                    proj_idx = (i * len(projections) // num_frames) % len(projections)
                    name, p_img, p_seg, p_mask = projections[proj_idx]
                    angle = (360.0 * i / num_frames) % 360.0

                    rotated_img = rotate(p_img, angle=angle, reshape=False, order=1, mode='nearest')
                    rotated_seg = rotate(p_seg, angle=angle, reshape=False, order=0, mode='nearest')
                    rotated_mask = rotate(p_mask, angle=angle, reshape=False, order=0, mode='nearest')
                    rotated_seg *= (rotated_mask > 0)

                    left_rgb = np.stack([rotated_img]*3, axis=-1)
                    right_rgb = overlay(rotated_img, rotated_seg, rotated_mask, alpha=0.6)

                    sep = np.ones((left_rgb.shape[0], 8, 3), dtype=np.float32)
                    composed = np.concatenate([left_rgb, sep, right_rgb], axis=1)

                    if fixed_size is None:
                        fixed_size = composed.shape
                    else:
                        h, w, c = composed.shape
                        H, W, _ = fixed_size
                        pad_h, pad_w = max(0, H - h), max(0, W - w)
                        if pad_h or pad_w:
                            composed = np.pad(composed, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
                        else:
                            composed = composed[:H, :W, :]

                    frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
                    plt.imsave(frame_path, composed)
                    frames.append(imageio.v2.imread(frame_path))

                print(f"Saving stable GIF to {output_path}...")
                imageio.mimsave(output_path, frames, duration=(1000 * 10 / num_frames), loop=0)
                print(f"✅ Fallback GIF saved successfully.")

        except Exception as e:
            print(f"❌ Fallback GIF creation failed: {e}")
            import traceback
            traceback.print_exc()
            

    def plot_static_montages(self, output_filename="static_montages.png", depths_pct=(0.25, 0.5, 0.75)):
        """
        Generates a robust "contact sheet" of 2D montages from all 3 planes.
        """
        print("Generating static 2D montage contact sheet...")
        output_path = os.path.join(self.output_dir, output_filename)
        num_depths = len(depths_pct)
        fig, axes = plt.subplots(3, num_depths * 3, figsize=(num_depths * 5, 15))
        fig.suptitle("Static 2D Slice Montages (Original | Ground Truth | Prediction)", fontsize=20, y=1.02)
        
        dims = self.original_image.shape
        planes = ['Axial', 'Coronal', 'Sagittal']
        plane_dims = [dims[2], dims[1], dims[0]] # Z, Y, X dims
        
        for i, (plane, plane_dim) in enumerate(zip(planes, plane_dims)):
            for j, depth_pct in enumerate(depths_pct):
                slice_index = int(plane_dim * depth_pct)
                ax_triplet = axes[i, j*3:j*3+3]
                self.plot_2d_montage(
                    slice_index,
                    plane=plane.lower(),
                    ax=ax_triplet
                )
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Static montages saved to {output_path}")


    def create_slice_scroll_gif(self, plane='axial', output_filename="axial_scroll.gif", fps=10, step=1):
        """
        Generates a dynamic "scroll" GIF, replacing the broken ipywidgets slider.
        """
        print(f"Generating 2D {plane} scroll GIF (in-memory)...")
        output_filename = output_filename.replace('.gif', f'_{plane}.gif')
        output_path = os.path.join(self.output_dir, output_filename)
        
        if plane == 'axial':
            num_slices = self.original_image.shape[2]
        elif plane == 'coronal':
            num_slices = self.original_image.shape[1]
        elif plane == 'sagittal':
            num_slices = self.original_image.shape[0]
        else:
            raise ValueError(f"Unknown plane: {plane}")

        frames = []
        for i in range(0, num_slices, step):
            fig = self.plot_2d_montage(i, plane=plane, ax=None)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.imread(buf))

        print(f"Saving {plane} scroll GIF to {output_path}...")
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"2D {plane} scroll GIF creation complete.")