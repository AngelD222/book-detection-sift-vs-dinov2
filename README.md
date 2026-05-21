# book-detection-sift-vs-dinov2
Computer Vision system that locates books on cluttered shelves by comparing classical methods (SIFT, ORB, RANSAC) against Deep Learning (DINOv2, SuperPoint + LightGlue), with custom NMS and AR wireframe overlay via homography


The Jupyter notebook contains all pre-rendered visual outputs and heatmaps. You can inspect the complete pipeline or run the code interactively directly in Google Colab via the following link:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uHEQYoa1B5vWRf07dGzqFDQoRB2XJQsQ#scrollTo=XZw813JiwDbR)


# Book Detection on Cluttered Shelves: Classical CV vs. Deep Learning

Computer Vision system that locates specific books on a shelf from a reference image of the spine, comparing **classical methods** (SIFT, ORB + RANSAC) against **Deep Learning** (SuperPoint + LightGlue, DINOv2 with a custom NMS). Includes an **AR-style wireframe overlay** computed via geometric homography.

> Academic project developed as part of the *Computer Vision* course of the MSc in Artificial Intelligence (Universidad de Murcia, 2025-2026), using a custom dataset of shelves and book spines captured by the team.

---

## Highlights

- **Robustness benchmark** of SIFT and ORB against rotation, scale and low-light transformations.
- **Homography estimation with RANSAC** to localize books on a shelf, including challenging cases: zoom, tilt and occlusion.
- **Custom Non-Maximum Suppression (NMS)** algorithm to extract multiple peaks from DINOv2 cosine-similarity heatmaps.
- **AR overlay** projecting the Canny edges of the original spine onto the detected book using `warpPerspective`.
- **Side-by-side evaluation** of three matching pipelines:
  - `SIFT + RANSAC`
  - `SuperPoint + LightGlue + RANSAC`
  - `DINOv2 patch features + cosine similarity + custom NMS`

---

## Key Findings

| Scenario | Best Method | Notes |
|---|---|---|
| Close-up shelf, clean spines | DINOv2 (NMS) | Locates almost every book correctly |
| Tilted shelf / perspective change | DINOv2 (NMS) | Wins thanks to semantic features |
| Repetitive geometric patterns | DINOv2 (NMS) | SIFT confuses ambiguous keypoints |
| Spine images with clutter (hands, background) | SIFT + RANSAC | Classical features isolate the object better |
| Heavy occlusion | DINOv2 (NMS) | LightGlue fails (too few visible local matches) |

**Bottom line:** DINOv2 is more robust under geometric distortion and difficult viewpoints, but more sensitive to noisy reference images. Classical pipelines remain a fast, tolerant baseline when the query spine is dirty or partially visible.

---

## Project Structure

The notebook follows the structure of the course assignment:

- **BT3a** — Local detector/descriptor comparison (SIFT vs ORB) under rotation, scale and illumination changes.
- **BT3b** — Feature matching with `BFMatcher` + Lowe's ratio test.
- **BT3c** — Robust homography estimation with RANSAC; green bounding box of the detected book.
- **BT3f** — Deep Learning pipelines: SuperPoint + LightGlue, DINOv2 with custom NMS.
- **BT3g** — AR-style overlay: Canny edges of the spine warped onto the shelf via the estimated homography.
- **BT3i** — Corner-based matching (Harris, ANMS) as a baseline.

```
.
├── notebooks/
│   └── VA_BT3_encontrar_libros.ipynb
├── data/                      # (custom dataset — see note below)
│   ├── lomos/                 # book spines (queries)
│   └── estanteria/            # shelf photos (targets)
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **Languages:** Python
- **Classical CV:** OpenCV (SIFT, ORB, BFMatcher, RANSAC, `findHomography`, `warpPerspective`, Canny)
- **Deep Learning:** PyTorch, Kornia, [LightGlue](https://github.com/cvg/LightGlue), [DINOv2](https://github.com/facebookresearch/dinov2)
- **Utilities:** NumPy, SciPy, Matplotlib
