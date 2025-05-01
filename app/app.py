import os
import sys
import time

import numpy as np
import cv2
from PIL import Image
import torch
from flask import Flask, render_template, request, redirect, url_for
from skimage.metrics import peak_signal_noise_ratio
from werkzeug.utils import secure_filename

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, SRC_ROOT)

from faces.model   import ColorNet as FaceNet
from general.model import ColorNet as GenNet
from faces.predict import preprocess, postprocess, inpaint_ab

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(os.path.join(os.path.dirname(__file__), UPLOAD_FOLDER), exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED = {'png','jpg','jpeg','bmp'}
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','models'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

face_model = FaceNet().to(device)
face_model.load_state_dict(torch.load(
    os.path.join(MODEL_DIR,'faces.pt'), map_location=device
))
face_model.eval()

gen_model = GenNet().to(device)
gen_model.load_state_dict(torch.load(
    os.path.join(MODEL_DIR,'general.pt'), map_location=device
))
gen_model.eval()

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='POST':
        f      = request.files.get('image')
        domain = request.form.get('domain','face')
        if not f or not allowed_file(f.filename):
            return redirect(request.url)

        fname       = secure_filename(f.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        f.save(upload_path)

        orig_pil, L = preprocess(upload_path, size=128)
        L = L.to(device)

        with torch.no_grad():
            ab_pred = gen_model(L) if domain=='general' else face_model(L)

        ab_before = ab_pred.detach().cpu().squeeze(0).permute(1,2,0).numpy()

        ab_pred, pct_inpainted = inpaint_ab(ab_pred, device, return_pct=True)

        ab_after = ab_pred.detach().cpu().squeeze(0).permute(1,2,0).numpy()

        mse_ab   = np.mean((ab_before - ab_after)**2)
        psnr_ab  = 10 * np.log10(4.0 / mse_ab) if mse_ab>0 else float('inf')

        out_pil = postprocess(L, ab_pred)
        out_fname = f"out_{domain}_{fname}"
        out_path  = os.path.join(app.config['UPLOAD_FOLDER'], out_fname)
        out_pil.save(out_path)

        out_np      = np.array(out_pil).astype(np.float32)/255.0
        mean_rgb    = (out_np.reshape(-1,3).mean(axis=0)*255).astype(int)
        dominant_hex = '#%02x%02x%02x' % tuple(mean_rgb.tolist())

        hsv       = cv2.cvtColor((out_np*255).astype('uint8'), cv2.COLOR_RGB2HSV)
        hist, _   = np.histogram(hsv[:,:,0], bins=36, range=(0,180), density=True)
        hist     += 1e-8
        color_entropy = float(-np.sum(hist * np.log2(hist)))

        return render_template('result.html',
            model_used     = 'general.pt' if domain=='general' else 'faces.pt',
            pct_inpainted  = round(pct_inpainted,1),
            psnr_ab        = round(psnr_ab,2),
            size           = out_pil.size[0],
            dominant_color = dominant_hex,
            color_entropy  = round(color_entropy,3),
            original       = url_for('static', filename=f'uploads/{fname}'),
            result         = url_for('static', filename=f'uploads/{out_fname}')
        )

    return render_template('index.html')
