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
from skimage.morphology import remove_small_objects, closing, disk
from tqdm import tqdm


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


def process_image(path: Path, clahe_path: Path, out_path: Path, kirsch_path: Path, overwrite: bool = False) -> None:
	if out_path.exists() and not overwrite:
		logging.debug("Skipping existing: %s", out_path)
		return
	
    # Ensure parent exists
	out_path.parent.mkdir(parents=True, exist_ok=True)
	clahe_path.parent.mkdir(parents=True, exist_ok=True)

    # Read and process
	img = read_image(path)
	green = get_green_channel(img)

	# Apply a slight blur to reduce noise (median)
	green_blur = cv2.medianBlur(green, 3)

	clahe = apply_clahe(green_blur)
	cv2.imwrite(str(clahe_path), clahe)

	edge_img = apply_kirsch(clahe)
	cv2.imwrite(str(out_path), edge_img)

	# enhanced = enhance_vessels(clahe)

	# mask = binarize_and_postprocess(enhanced)

	
	# Save as PNG 0/255
	# cv2.imwrite(str(out_path), clahe)


def iter_images(input_dir: Path):
	for p in sorted(input_dir.glob("**/*")):
		if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
			yield p




def main() -> None:
	parser = argparse.ArgumentParser(description="Generate vessel segmentation masks from fundus images")
	parser.add_argument("--input-dir", type=Path, default=Path("data/raw"), help="Directory with input images")
	parser.add_argument("--clahe-dir", type=Path, default=Path("data/clahe_enhanced"), help="Directory to write clahe images")
	parser.add_argument("--kirsch-dir", type=Path, default=Path("data/kirsch_enhanced"), help="Directory to write kirsch images")
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
		clahe_p = clahe_p.with_suffix(".png")
		out_p = out_p.with_suffix(".png")
		kirsch_p = kirsch_p.with_suffix(".png")
		
		try:
			process_image(p, clahe_p, out_p, kirsch_p, overwrite=args.overwrite)
		except Exception as e:
			logging.warning("Failed processing %s: %s", p, e)


if __name__ == "__main__":
	main()

