#!/usr/bin/env python3
"""
generate_vessel_mask.py

Pipeline per estrarre maschere dei vasi sanguigni da immagini fundus presenti in `data/raw`.

Principali passaggi implementati:
- leggere l'immagine (cv2)
- usare il canale verde (migliore contrasto per i vasi)
- equalizzazione adattiva del contrasto (CLAHE)
- filtro Frangi per esaltare strutture tubolari (vasi)
- sogliatura con Otsu
- operazioni morfologiche e rimozione di piccoli oggetti
- salvare la maschera binaria (0/255) in `data/vessel_mask`

Contratto (input/output):
- input: directory con immagini (png/jpg). Si leggono immagini RGB/gray.
- output: directory dove vengono scritte le maschere con lo stesso nome file ma nella cartella di output.

Error modes e comportamento:
- se un file non è leggibile viene saltato con warning
- se la directory di output non esiste viene creata
- opzione --overwrite per sovrascrivere maschere esistenti

Requisiti minimi (aggiungere a requirements.txt):
opencv-python, numpy, scikit-image, tqdm

Esempio d'uso:
	python src/generate_vessel_mask.py --input-dir data/raw --output-dir data/vessel_mask

"""

from __future__ import annotations

import argparse
from html import parser
import logging
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from skimage import exposure, img_as_ubyte
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import remove_small_objects, closing, disk, opening
from tqdm import tqdm
from fcmeans import FCM
from active_contour import region_seg


def read_image(path: Path) -> np.ndarray:
	img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
	if img is None:
		raise IOError(f"Cannot read image: {path}")
	# If grayscale, keep as single channel; else use BGR
	return img


def get_green_channel(img: np.ndarray) -> np.ndarray:
	# If color image, OpenCV loads as BGR
	if img.ndim == 3 and img.shape[2] >= 2:
		green = img[:, :, 1]
	else:
		green = img if img.ndim == 2 else img[:, :, 0]
	return green


def apply_clahe(channel: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
	clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
	return clahe.apply(channel)


def enhance_vessels(image_gray: np.ndarray) -> np.ndarray:
	"""Return a vessel-enhanced float image in range [0,1]."""
	# Convert to float in [0,1]
	image_f = image_gray.astype("float32")
	if image_f.max() > 0:
		image_f /= image_f.max()

	# Frangi filter expects float image
	fr = frangi(image_f)
	# rescale to [0,1]
	fr = exposure.rescale_intensity(fr, in_range='image', out_range=(0.0, 1.0))
	return fr


def binarize_and_postprocess(enhanced: np.ndarray, min_size: int = 200, closing_radius: int = 2) -> np.ndarray:
	# Otsu threshold
	try:
		th = threshold_otsu(enhanced)
	except Exception:
		th = 0.5
	bw = enhanced >= th

	# Morphological closing to join broken vessels
	bw_closed = closing(bw, disk(closing_radius))

	# Remove small objects
	bw_clean = remove_small_objects(bw_closed, min_size=min_size)

	# Convert to uint8 0/255
	out = img_as_ubyte(bw_clean)
	return out

def getKirschFilters():
    kirsch = [5, -3, -3, -3, -3, -3, 5, 5]
    rot = lambda l, n: l[-n:] + l[:-n]
    filts = np.zeros((8, 3, 3), dtype=np.int32)
    for d in range(8):
        filts[d] = np.array([kirsch[0:3],
                             [kirsch[7], 0, kirsch[3]],
                             kirsch[6:3:-1]], dtype=np.int32)
        kirsch = rot(kirsch, 1)
    return filts

def apply_kirsch(image: np.ndarray) -> np.ndarray:
	filters = getKirschFilters()
	rows, cols = image.shape
	edge_image = np.zeros((rows, cols), dtype=np.float32)

	# Apply each filter and keep the maximum response
	for f in filters:
		filtered = cv2.filter2D(image.astype(np.float32), -1, f)
		edge_image = np.maximum(edge_image, filtered)

	# Normalize to range [0, 255]
	edge_image = cv2.normalize(edge_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	return edge_image.astype(np.uint8)

def apply_fcm(image: np.ndarray, n_clusters: int = 2) -> np.ndarray:
	# Reshape image to (num_pixels, 1)
	pixels = image.reshape(-1, 1).astype(np.float32)

	# Apply FCM
	fcm = FCM(n_clusters=n_clusters, m=2.0, max_iter=100, error=1e-5)
	fcm.fit(pixels)

	# Get cluster centers and labels
	centers = fcm.centers.flatten()
	labels = fcm.predict(pixels).astype(int).reshape(image.shape)

	# Create segmented image based on cluster centers (for debugging/visualization)
	segmented = np.zeros_like(image, dtype=np.uint8)
	for i, center in enumerate(centers):
		segmented[labels == i] = int(center)

	# --- Inizio traduzione del blocco MATLAB per calcolare la maschera I2 ---
	# Basato sullo script fornito: clustering iterativo a 2 cluster + apertura morfologica
	IM = image.astype(np.float64)
	maxX, maxY = IM.shape
	IMM = np.stack((IM, IM), axis=2)

	cc1 = 8.0
	cc2 = 250.0
	ttFcm = 0

	IMMM = np.zeros_like(IM, dtype=np.float64)
	IX2 = np.ones_like(IM, dtype=np.int32)
	vessel_mask = None

	for _ in range(20):
		ttFcm += 1
		c1 = np.full((maxX, maxY), cc1, dtype=np.float64)
		c2 = np.full((maxX, maxY), cc2, dtype=np.float64)
		c = np.stack((c1, c2), axis=2)

		ree = np.full((maxX, maxY), 1e-6, dtype=np.float64)
		ree1 = np.stack((ree, ree), axis=2)

		distance = IMM - c
		distance = distance * distance + ree1
		daoShu = 1.0 / distance
		daoShu2 = daoShu[:, :, 0] + daoShu[:, :, 1]

		distance1 = distance[:, :, 0] * daoShu2
		# avoid division by zero
		distance1[distance1 == 0] = np.finfo(float).eps
		u1 = 1.0 / distance1

		distance2 = distance[:, :, 1] * daoShu2
		distance2[distance2 == 0] = np.finfo(float).eps
		u2 = 1.0 / distance2

		# cluster centers update
		num1 = np.sum((u1 * u1) * IM)
		den1 = np.sum(u1 * u1) + np.finfo(float).eps
		ccc1 = num1 / den1

		num2 = np.sum((u2 * u2) * IM)
		den2 = np.sum(u2 * u2) + np.finfo(float).eps
		ccc2 = num2 / den2

		tmpMatrix = np.array([abs(cc1 - ccc1) / (abs(cc1) + np.finfo(float).eps),
							  abs(cc2 - ccc2) / (abs(cc2) + np.finfo(float).eps)])

		# hard assignment like in the MATLAB code
		IX2 = np.where(u1 >= u2, 1, 2)

		if np.max(tmpMatrix) < 1e-4:
			# compute current IMMM and mask at convergence and break
			IMMM = np.where(IX2 == 2, 254.0, 8.0)
			background = opening(IMMM, disk(45))
			I2 = IMMM - background
			I2_bool = remove_small_objects((I2 > 0), min_size=50)
			vessel_mask = I2_bool
			break
		else:
			cc1 = ccc1
			cc2 = ccc2

		# construct IMMM per iter (not strictly needed each iter, keep for parity)
		IMMM = np.where(IX2 == 2, 254.0, 8.0)
		# background estimation with large opening
		background = opening(IMMM, disk(45))
		I2 = IMMM - background
		# remove small objects (bwareaopen) -> work on boolean mask
		I2_bool = remove_small_objects((I2 > 0), min_size=50)

	# final IMMM/IX2 after convergence
	# If loop finished without setting vessel_mask (no early convergence), compute final mask
	if vessel_mask is None:
		IMMM = np.where(IX2 == 2, 200.0, 1.0)
		background = opening(IMMM, disk(45))
		I2 = IMMM - background
		vessel_mask = remove_small_objects((I2 > 0), min_size=50)

	# Return segmented image, labels matrix, centers and the matlab-style mask (boolean)
	return segmented, labels, centers, vessel_mask


def process_image(path: Path, clahe_path: Path, out_path: Path, kirsch_path: Path, fuzzy_path: Path, overwrite: bool = False) -> None:
	if out_path.exists() and not overwrite:
		logging.debug("Skipping existing: %s", out_path)
		return
	# Ensure parent exists
	out_path.parent.mkdir(parents=True, exist_ok=True)
	clahe_path.parent.mkdir(parents=True, exist_ok=True)
	kirsch_path.parent.mkdir(parents=True, exist_ok=True)
	fuzzy_path.parent.mkdir(parents=True, exist_ok=True)

	# Read and process
	img = read_image(path)
	green = get_green_channel(img)

	# Apply a slight blur to reduce noise (median)
	green_blur = cv2.medianBlur(green, 3)

	print("Processing CLAHE enhancement...")
	clahe = apply_clahe(green_blur)
	cv2.imwrite(str(clahe_path), clahe)
	print("Processing Kirsch edge enhancement...")
	edge_img = apply_kirsch(clahe)
	edge_img = closing(edge_img, disk(1))
	cv2.imwrite(str(kirsch_path), edge_img)
	
	print("Processing FCM clustering...")
	segmented, labels, centers, vessel_mask = apply_fcm(edge_img, n_clusters=2)
	cv2.imwrite(str(fuzzy_path), segmented)
	
	print("Processing region-based active contour segmentation...")
	# use vessel_mask (from MATLAB-style routine) as initial mask for region_seg
	init_mask = vessel_mask.astype(bool)
	try:
		seg_mask = region_seg(clahe, init_mask, max_its=300, alpha=0.2, display=False)
	except Exception as e:
		logging.warning("region_seg failed for %s: %s", path, e)
		seg_mask = init_mask

	# save final segmentation to out_path (0/255)
	out_mask = (seg_mask.astype(np.uint8) * 255)
	cv2.imwrite(str(out_path), out_mask)

def iter_images(input_dir: Path):
	for p in sorted(input_dir.glob("**/*")):
		if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
			yield p




def main() -> None:
	parser = argparse.ArgumentParser(description="Generate vessel segmentation masks from fundus images")
	parser.add_argument("--input-dir", type=Path, default=Path("data/raw"), help="Directory with input images")
	parser.add_argument("--clahe-dir", type=Path, default=Path("data/clahe_enhanced"), help="Directory to write clahe images")
	parser.add_argument("--kirsch-dir", type=Path, default=Path("data/kirsch_enhanced"), help="Directory to write kirsch images")
	parser.add_argument("--fuzzy-dir", type=Path, default=Path("data/fuzzy_segmentation"), help="Directory to write fuzzy segmented images")
	parser.add_argument("--output-dir", type=Path, default=Path("data/vessel_mask"), help="Directory to write masks")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks")
	parser.add_argument("--min-size", type=int, default=200, help="Minimum connected component size to keep (pixels)")
	parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bar")
	parser.add_argument("--verbose", action="store_true")
	args = parser.parse_args()

	logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

	if not args.input_dir.exists():
		logging.error("Input directory does not exist: %s", args.input_dir)
		return

	images = list(iter_images(args.input_dir))
	if not images:
		logging.info("No images found in %s", args.input_dir)
		return

	iterator = images
	if args.progress:
		iterator = tqdm(images, desc="Processing images", unit="img")

	for p in iterator:
		try:
			rel = p.relative_to(args.input_dir)
		except Exception:
			rel = p.name
		out_p = args.output_dir / rel
		clahe_p = args.clahe_dir / rel
		kirsch_p = args.kirsch_dir / rel
		fuzzy_p = args.fuzzy_dir / rel
		clahe_p = clahe_p.with_suffix(".png")
		out_p = out_p.with_suffix(".png")
		kirsch_p = kirsch_p.with_suffix(".png")
		fuzzy_p = fuzzy_p.with_suffix(".png")

		try:
			process_image(p, clahe_p, out_p, kirsch_p, fuzzy_p, overwrite=args.overwrite)
		except Exception as e:
			logging.warning("Failed processing %s: %s", p, e)


if __name__ == "__main__":
	main()
