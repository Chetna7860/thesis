"""
============================================================
KAGGE NOTEBOOK: Save outputs, zip, and download
============================================================
Copy and run each cell below in order on Kaggle.
"""

# ============================================================
# CELL 1: Free space + organize outputs into final_outputs/
# ============================================================
import os, shutil, glob, zipfile, json
from datetime import datetime

# Free space by removing bulky per-epoch checkpoints
print("Freeing space...")
ckpt_files = glob.glob('/kaggle/working/thesis/outputs/checkpoints/**/*', recursive=True)
removed = 0
for f in ckpt_files:
    if os.path.isfile(f) and 'best' not in os.path.basename(f).lower():
        os.remove(f)
        removed += 1
print(f"Removed {removed} per-epoch checkpoints")

# Remove old zips
for old in glob.glob('/kaggle/working/*.zip'):
    os.remove(old)

# Create final_outputs/
FINAL = '/kaggle/working/final_outputs'
os.makedirs(FINAL, exist_ok=True)

def collect(src_pat, dest_subdir, base='/kaggle/working/thesis', flat=True):
    files = glob.glob(f'{base}/{src_pat}', recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    d = os.path.join(FINAL, dest_subdir)
    os.makedirs(d, exist_ok=True)
    for f in files:
        if flat:
            shutil.copy2(f, os.path.join(d, os.path.basename(f)))
        else:
            rel = os.path.relpath(f, f'{base}/outputs')
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            shutil.copy2(f, p)
    return files

# Models (final + best only)
m = collect('outputs/models/*', 'models')
print(f"Models: {len(m)}")

# Plots
p = collect('outputs/plots/*', 'graphs')
print(f"Graphs/plots: {len(p)}")

# Logs (preserve structure)
l = collect('outputs/logs/**/*', 'logs', flat=False)
print(f"Log files: {len(l)}")

# GradCAM
g = collect('outputs/gradcam/*', 'gradcam')
print(f"Grad-CAM: {len(g)}")

# JSON results
j = collect('outputs/**/*.json', 'results')
print(f"JSON results: {len(j)}")

# Config
c = collect('config/*.yaml', 'config', flat=True)
print(f"Config files: {len(c)}")

# Save summary
summary = {
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'models': len(m), 'graphs': len(p), 'logs': len(l),
    'gradcam': len(g), 'results': len(j), 'configs': len(c)
}
with open(os.path.join(FINAL, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

total_mb = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fn in os.walk(FINAL) for f in fn) / 1048576
print(f"\nTotal: {total_mb:.1f} MB in {FINAL}")


# ============================================================
# CELL 2: Zip final_outputs/ -> thesis_results.zip
# ============================================================
ZIP = '/kaggle/working/thesis_results.zip'
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(FINAL):
        for file in files:
            fp = os.path.join(root, file)
            zf.write(fp, os.path.relpath(fp, FINAL))

sz = os.path.getsize(ZIP) / 1048576
cnt = len(zipfile.ZipFile(ZIP).namelist())
print(f"✓ Created: {ZIP}")
print(f"  Size: {sz:.1f} MB | Files: {cnt}")


# ============================================================
# CELL 3: Click the link below to download
# ============================================================
from IPython.display import FileLink
FileLink('/kaggle/working/thesis_results.zip')
