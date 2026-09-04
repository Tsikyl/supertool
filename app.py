import os
import tempfile
import zipfile
import numpy as np
import cv2
from flask import Flask, request, send_file, render_template
from PIL import Image, ImageEnhance, ImageFilter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, A3, A2, A1, landscape
from reportlab.lib.units import mm

app = Flask(__name__)

FORMATS = {'A4': A4, 'A3': A3, 'A2': A2, 'A1': A1}

def trouver_emprise_intelligente(img_np):
    # Utilisation d'OpenCV pour filtrer le bruit (textes, cartouches)
    _, binaire = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Dilatation pour relier les murs fragmentés et ignorer les petits textes
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(binaire, kernel, iterations=3)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, img_np.shape[1]
        
    # Filtrer les petits contours (bruit) et trouver la boîte englobante globale des grands contours
    x_min, x_max = img_np.shape[1], 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000: # Ignorer les petites taches
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            x_max = max(x_max, x + w)
            
    return x_min, x_max

def traiter_plan(chemin_image, chemin_pdf, largeur_reelle_m, echelle, format_page):
    img = Image.open(chemin_image).convert('L')
    img_amelioree = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    img_amelioree = ImageEnhance.Contrast(img_amelioree).enhance(1.8)
    img_amelioree = img_amelioree.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    chemin_temp_jpg = chemin_pdf + "_temp.jpg"
    img_amelioree.convert('RGB').save(chemin_temp_jpg, "JPEG", quality=85)
    
    img_np = np.array(img)
    gauche_px, droite_px = trouver_emprise_intelligente(img_np)
    largeur_pixels_amelioree = (droite_px - gauche_px) * 2

    largeur_cible_mm = (float(largeur_reelle_m) * 1000) / int(echelle)
    mm_par_pixel = largeur_cible_mm / largeur_pixels_amelioree if largeur_pixels_amelioree > 0 else 1
    largeur_img_pdf_mm, hauteur_img_pdf_mm = img_amelioree.width * mm_par_pixel, img_amelioree.height * mm_par_pixel

    taille_page = landscape(FORMATS.get(format_page, A3))
    c = canvas.Canvas(chemin_pdf, pagesize=taille_page)
    c.drawImage(chemin_temp_jpg, (taille_page[0] - (largeur_img_pdf_mm * mm)) / 2.0, (taille_page[1] - (hauteur_img_pdf_mm * mm)) / 2.0, width=largeur_img_pdf_mm * mm, height=hauteur_img_pdf_mm * mm)
    c.save()
    os.remove(chemin_temp_jpg)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_batch', methods=['POST'])
def upload_batch():
    fichiers = request.files.getlist('files[]')
    largeur = request.form.get('largeur', 9.0)
    echelle = request.form.get('echelle', 50)
    format_page = request.form.get('format', 'A3')

    if not fichiers or fichiers[0].filename == '':
        return "Aucun fichier", 400

    fd_zip, path_zip = tempfile.mkstemp(suffix=".zip")
    os.close(fd_zip)

    with zipfile.ZipFile(path_zip, 'w') as zipf:
        for file in fichiers:
            fd_img, path_img = tempfile.mkstemp(suffix=".png")
            fd_pdf, path_pdf = tempfile.mkstemp(suffix=".pdf")
            os.close(fd_img)
            os.close(fd_pdf)
            
            file.save(path_img)
            traiter_plan(path_img, path_pdf, largeur, echelle, format_page)
            
            nom_base = os.path.splitext(file.filename)[0]
            zipf.write(path_pdf, f"{nom_base}_Echelle_{echelle}_{format_page}.pdf")
            
            os.remove(path_img)
            os.remove(path_pdf)

    return send_file(path_zip, as_attachment=True, download_name="Plans_Architecture_Batch.zip")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
