import os
import sys
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch

# ─── Make sure we can import from src/ ─────────────────────────────────────────
# file is: project_root/app/app.py
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from faces.model   import ColorNet as FaceNet
from general.model import ColorNet as GenNet
from faces.predict import preprocess, postprocess, inpaint_ab

# ─── Flask setup ───────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(os.path.join(os.path.dirname(__file__), UPLOAD_FOLDER), exist_ok=True)

ALLOWED_EXTS  = {'png','jpg','jpeg','bmp'}
MODEL_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Load models once at startup ───────────────────────────────────────────────
face_model = FaceNet().to(device)
face_model.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, 'faces.pt'),
    map_location=device
))
face_model.eval()

gen_model = GenNet().to(device)
gen_model.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, 'general.pt'),
    map_location=device
))
gen_model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTS

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        file   = request.files.get('image')
        domain = request.form.get('domain', 'face')
        if not file or not allowed_file(file.filename):
            return redirect(request.url)

        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # Preprocess image
        _, L = preprocess(upload_path, size=256)
        L = L.to(device)

        # Model inference
        with torch.no_grad():
            if domain == 'general':
                ab_pred = gen_model(L)
            else:
                ab_pred = face_model(L)

        # Inpaint holes for portraits
        if domain == 'face':
            ab_pred = inpaint_ab(ab_pred, device)

        # Postprocess & save output
        out_img      = postprocess(L, ab_pred)
        out_filename = f"out_{domain}_{filename}"
        out_path     = os.path.join(app.config['UPLOAD_FOLDER'], out_filename)
        out_img.save(out_path)

        return render_template('result.html',
                               original = url_for('static', filename=f'uploads/{filename}'),
                               result   = url_for('static', filename=f'uploads/{out_filename}'))

    return render_template('index.html')

