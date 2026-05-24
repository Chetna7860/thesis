"""
Kaggle notebook code to save all outputs, zip, and upload to Google Drive.

Instructions:
1. Create OAuth client ID at https://console.cloud.google.com/apis/credentials
   (Enable Drive API first at https://console.cloud.google.com/apis/library/drive.googleapis.com)
2. Download client_secrets.json
3. On Kaggle: Settings -> Secrets -> Add new secret
   Name: GDRIVE_CLIENT_SECRETS
   Value: (paste the ENTIRE contents of client_secrets.json)
4. Run these cells in order.
"""

# ============================================================
# CELL 1: Organize all outputs into final_outputs/
# ============================================================
import os, shutil, glob, zipfile, json
from datetime import datetime

print("=" * 60)
print("Step 1: Organizing all outputs into final_outputs/")
print("=" * 60)

FINAL_DIR = '/kaggle/working/final_outputs'
os.makedirs(FINAL_DIR, exist_ok=True)

# Find all source outputs
source_dirs = [
    '/kaggle/working/thesis/outputs',
    '/kaggle/working/thesis/outputs/plots',
    '/kaggle/working/thesis/outputs/models',
    '/kaggle/working/thesis/outputs/logs',
    '/kaggle/working/thesis/outputs/checkpoints',
    '/kaggle/working/thesis/outputs/gradcam',
]

# ---- Category 1: Trained models (.pth, .h5, .weights.h5) ----
model_files = []
for pat in ['**/*.pth', '**/*.h5', '**/*.weights.h5']:
    model_files.extend(glob.glob(f'/kaggle/working/thesis/outputs/**/{pat}', recursive=True))

# Keep only final models and best checkpoints (not every epoch)
final_models = [f for f in model_files if 'final' in f.lower()]
best_models = [f for f in model_files if 'best' in f.lower() and f not in final_models]

models_dir = os.path.join(FINAL_DIR, 'models')
os.makedirs(models_dir, exist_ok=True)
for f in final_models + best_models:
    shutil.copy2(f, os.path.join(models_dir, os.path.basename(f)))
    print(f"  Copied model: {os.path.basename(f)}")

# ---- Category 2: Plots and graphs ----
plot_files = glob.glob('/kaggle/working/thesis/outputs/plots/*')
plots_dir = os.path.join(FINAL_DIR, 'graphs')
os.makedirs(plots_dir, exist_ok=True)
for f in plot_files:
    shutil.copy2(f, os.path.join(plots_dir, os.path.basename(f)))
    print(f"  Copied graph: {os.path.basename(f)}")

# ---- Category 3: Log files ----
log_files = glob.glob('/kaggle/working/thesis/outputs/logs/**/*', recursive=True)
log_files = [f for f in log_files if os.path.isfile(f)]
logs_dir = os.path.join(FINAL_DIR, 'logs')
os.makedirs(logs_dir, exist_ok=True)
for f in log_files:
    rel = os.path.relpath(f, '/kaggle/working/thesis/outputs/logs')
    dest = os.path.join(logs_dir, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(f, dest)
print(f"  Copied {len(log_files)} log files")

# ---- Category 4: Grad-CAM images ----
gradcam_files = glob.glob('/kaggle/working/thesis/outputs/gradcam/*')
gradcam_dir = os.path.join(FINAL_DIR, 'gradcam')
os.makedirs(gradcam_dir, exist_ok=True)
for f in gradcam_files:
    shutil.copy2(f, os.path.join(gradcam_dir, os.path.basename(f)))
print(f"  Copied {len(gradcam_files)} Grad-CAM images")

# ---- Category 5: JSON results ----
json_files = glob.glob('/kaggle/working/thesis/outputs/**/*.json', recursive=True)
results_dir = os.path.join(FINAL_DIR, 'results')
os.makedirs(results_dir, exist_ok=True)
for f in json_files:
    shutil.copy2(f, os.path.join(results_dir, os.path.basename(f)))
    print(f"  Copied result: {os.path.basename(f)}")

# ---- Category 6: Prediction results (from predict.py) ----
pred_files = glob.glob('/kaggle/working/thesis/*.json')
pred_files += glob.glob('/kaggle/working/thesis/predictions.*')
pred_dir = os.path.join(FINAL_DIR, 'predictions')
os.makedirs(pred_dir, exist_ok=True)
for f in pred_files:
    shutil.copy2(f, os.path.join(pred_dir, os.path.basename(f)))
    print(f"  Copied prediction: {os.path.basename(f)}")

# ---- Category 7: Config used ----
config_files = glob.glob('/kaggle/working/thesis/config/*.yaml')
config_dir = os.path.join(FINAL_DIR, 'config')
os.makedirs(config_dir, exist_ok=True)
for f in config_files:
    shutil.copy2(f, os.path.join(config_dir, os.path.basename(f)))
    print(f"  Copied config: {os.path.basename(f)}")

# ---- Summary report ----
summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'models': len(final_models) + len(best_models),
    'graphs': len(plot_files),
    'logs': len(log_files),
    'gradcam': len(gradcam_files),
    'json_results': len(json_files),
    'predictions': len(pred_files),
}

with open(os.path.join(FINAL_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary: {json.dumps(summary, indent=2)}")
print(f"Total size: {sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(FINAL_DIR) for f in fn) / 1024 / 1024:.2f} MB")


# ============================================================
# CELL 2: Compress final_outputs/ into final_outputs.zip
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Compressing to final_outputs.zip")
print("=" * 60)

ZIP_PATH = '/kaggle/working/final_outputs.zip'

# Remove checkpoints to save space before zipping
checkpoint_dirs = glob.glob(f'{FINAL_DIR}/checkpoints/*')
if checkpoint_dirs:
    print("  Removing heavy checkpoints from zip (keeping only best)...")
    for cd in checkpoint_dirs:
        if os.path.isdir(cd):
            for f in glob.glob(os.path.join(cd, '*')):
                if 'best' not in os.path.basename(f).lower():
                    os.remove(f)
                    print(f"  Removed: {os.path.basename(f)}")

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(FINAL_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, FINAL_DIR)
            zf.write(file_path, arcname)

zip_size = os.path.getsize(ZIP_PATH) / 1024 / 1024
zinfo = zipfile.ZipFile(ZIP_PATH)
num_files = len(zinfo.namelist())
zinfo.close()

print(f"  Zip created: {ZIP_PATH}")
print(f"  Size: {zip_size:.2f} MB")
print(f"  Files inside: {num_files}")
print(f"  Contents: {', '.join(sorted(set(f.split('/')[0] for f in zipfile.ZipFile(ZIP_PATH).namelist())))}")


# ============================================================
# CELL 3: Upload to Google Drive
# Requires: Kaggle Secret 'GDRIVE_CLIENT_SECRETS' with OAuth JSON
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Uploading final_outputs.zip to Google Drive")
print("=" * 60)

try:
    from kaggle_secrets import UserSecretsClient

    secret_json = UserSecretsClient().get_secret("GDRIVE_CLIENT_SECRETS")

    if not secret_json:
        raise ValueError("GDRIVE_CLIENT_SECRETS secret not found or empty.")

    import json, io
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google_auth_oauthlib.flow import InstalledAppFlow

    SCOPES = ['https://www.googleapis.com/auth/drive']

    client_config = json.loads(secret_json)
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    creds = flow.run_local_server(port=0, open_browser=False)

    # Build Drive service
    service = build('drive', 'v3', credentials=creds)

    # Create or find target folder
    FOLDER_NAME = 'Kaggle_Outputs'
    query = f"name='{FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive').execute()
    items = results.get('files', [])

    if items:
        folder_id = items[0]['id']
        print(f"  Found existing folder: {FOLDER_NAME} (ID: {folder_id})")
    else:
        file_metadata = {
            'name': FOLDER_NAME,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        folder_id = folder.get('id')
        print(f"  Created folder: {FOLDER_NAME} (ID: {folder_id})")

    # Upload the zip
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_metadata = {
        'name': f'thesis_outputs_{timestamp}.zip',
        'parents': [folder_id]
    }
    media = MediaFileUpload(ZIP_PATH, mimetype='application/zip', resumable=True)

    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, name, size, webViewLink'
    ).execute()

    print(f"\n  Upload complete!")
    print(f"  File name: {uploaded['name']}")
    print(f"  File ID: {uploaded['id']}")
    print(f"  Size: {int(uploaded['size']) / 1024 / 1024:.2f} MB")
    print(f"  Drive link: {uploaded['webViewLink']}")

except ImportError:
    print("\n  ERROR: kaggle_secrets not available. Make sure you're running on Kaggle.")
    print("  Install required packages manually:")
    print("  !pip install google-auth-oauthlib google-api-python-client")
except ValueError as e:
    print(f"\n  ERROR: {e}")
    print("  To fix:")
    print("  1. Go to https://console.cloud.google.com/apis/credentials")
    print("  2. Enable Drive API: https://console.cloud.google.com/apis/library/drive.googleapis.com")
    print("  3. Create OAuth 2.0 Client ID (Desktop app)")
    print("  4. Download client_secrets.json")
    print("  5. On Kaggle: Settings -> Secrets -> Add secret 'GDRIVE_CLIENT_SECRETS'")
    print("  6. Paste the ENTIRE JSON content as the value")
except Exception as e:
    print(f"\n  Upload failed: {e}")
    print("  Fallback: Download the zip using FileLink instead.")
    from IPython.display import FileLink
    FileLink(ZIP_PATH)
