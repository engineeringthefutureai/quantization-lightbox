# Quantization Lightbox

> **Weight quantization, shown on pixels.**

An interactive visualizer that treats an image's color channels like a weight tensor and squeezes them into low bit-widths (1 to 8 bits), using schemes that directly parallel real LLM quantization formats (such as GGUF's Q4_0, Q4_K_M, GPTQ, and EXL2).

---

## Live Demo & GitHub Pages Setup

This repository is pre-configured to be served directly via **GitHub Pages**.

### Enabling GitHub Pages on GitHub

1. Go to your repository settings on GitHub: **Settings** &rarr; **Pages**.
2. Under **Build and deployment** &rarr; **Source**, choose one of the two options:
   - **GitHub Actions** *(Recommended)*: Uses the automated workflow included in [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) to deploy on every push to `main`.
   - **Deploy from a branch**: Select the `main` branch and `/ (root)` folder, then click **Save**.
3. Once deployed, your site will be live at:
   ```
   https://<your-username>.github.io/<repository-name>/
   ```

---

## Features

- **Interactive Quantization Controls**:
  - Adjust bit depth from 1 to 8 bits per value.
  - Change block size between 4×4, 8×8, and 16×16 tiles (matching GGUF block & super-block structures).
  - Toggle between decoded reconstructed image and error heatmap (`≥ 0.15` absolute error).
  - Use the built-in sample nebula image, load custom local images, or drag-and-drop any image file directly onto the page.
- **Quantization Schemes Implemented**:
  1. **Naive Uniform Grid**: Global per-tensor range, no metadata overhead (INT1 to INT8).
  2. **Block-Wise Scale**: Per-block fp16 minimum and scale (GGUF Q4_0, Q4_1, Q8_0 style).
  3. **Block + Error Diffusion**: Floyd–Steinberg error diffusion kernel during quantization (GPTQ counterpart).
  4. **Mixed Precision**: Dynamic bit-width allocation based on block detail/variance (Q4_K_M / EXL2 style).
  5. **Mixed + Diffusion**: Combines variable bit-widths with error diffusion.
- **Microscope & Inspection Tools**:
  - **Pixel Microscope**: Click any spot on the image to inspect an individual block channel-by-channel at the raw byte and bit level.
  - **Storage Bit Layout**: Shows exact binary packing of codes and IEEE 754 half-precision (fp16) metadata.
  - **Live Math Walkthrough**: Detailed step-by-step arithmetic (min, max, step size, clamp, round, dequantize, residual error).
  - **Cost Comparison Table**: Real-time storage consumption table (payload vs. metadata bits, bits/value, compression ratio).

---

## Local Development

Because this project is entirely self-contained with no build dependencies or package managers required, you can view it locally using any static HTTP server or by opening `index.html` in your browser.

Using Python:
```bash
python3 -m http.server 8000
```
Then navigate to `http://localhost:8000` in your browser.

Using Node.js:
```bash
npx serve .
```

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── deploy.yml            # Automated GitHub Actions deployment for Pages (triggers on main)
├── .nojekyll                     # Disables Jekyll processing on GitHub Pages
├── index.html                    # Main HTML5 entry point for GitHub Pages
├── style.css                     # Extracted modular stylesheet
├── app.js                        # Extracted application logic and quantizer engine
├── sample.png                    # Extracted standalone sample image asset
└── README.md                     # Project documentation
```

---

## License

MIT / Open Source — feel free to explore, fork, and use for learning and demonstrating quantization concepts.
