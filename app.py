import os
import tempfile
import numpy as np
from flask import Flask, request, send_file, render_template
from PIL import Image, ImageEnhance, ImageFilter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm

app = Flask(__name__)

def traiter_plan(chemin_image, chemin_pdf, largeur_reelle_m, echelle):
    img = Image.open(chemin_image).convert('L')
    img_amelioree = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    img_amelioree = ImageEnhance.Contrast(img_amelioree).enhance(1.8)
    img_amelioree = img_amelioree.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    chemin_temp_jpg = chemin_pdf + "_temp.jpg"
    img_amelioree.convert('RGB').save(chemin_temp_jpg, "JPEG", quality=85)
    
    binaire = np.array(img) < 100
    indices_colonnes = np.where(np.sum(binaire, axis=0) > 50)[0]
    largeur_pixels_amelioree = (indices_colonnes[-1] - indices_colonnes[0]) * 2

    largeur_cible_mm = (float(largeur_reelle_m) * 1000) / int(echelle)
    mm_par_pixel = largeur_cible_mm / largeur_pixels_amelioree
    largeur_img_pdf_mm, hauteur_img_pdf_mm = img_amelioree.width * mm_par_pixel, img_amelioree.height * mm_par_pixel

    largeur_page, hauteur_page = landscape(A3)
    c = canvas.Canvas(chemin_pdf, pagesize=landscape(A3))
    c.drawImage(chemin_temp_jpg, (largeur_page - (largeur_img_pdf_mm * mm)) / 2.0, (hauteur_page - (hauteur_img_pdf_mm * mm)) / 2.0, width=largeur_img_pdf_mm * mm, height=hauteur_img_pdf_mm * mm)
    c.save()
    os.remove(chemin_temp_jpg)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    largeur = request.form.get('largeur', 9.0)
    echelle = request.form.get('echelle', 50)

    if not file:
        return "Aucun fichier", 400

    fd_img, path_img = tempfile.mkstemp(suffix=".png")
    fd_pdf, path_pdf = tempfile.mkstemp(suffix=".pdf")
    os.close(fd_img)
    os.close(fd_pdf)

    file.save(path_img)
    traiter_plan(path_img, path_pdf, largeur, echelle)
    os.remove(path_img)

    return send_file(path_pdf, as_attachment=True, download_name="Plan_A3.pdf")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
