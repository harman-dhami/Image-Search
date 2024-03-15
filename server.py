import numpy as np
from PIL import Image
from features import Features
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

fea = Features()
feature = []
img_paths = []
for feature_path in Path("static/feature").glob("*.npy"):
    feature.append(np.load(feature_path))
    img_paths.append(Path("static/random_images") / (feature_path.stem + ".jpg"))
feature = np.array(feature)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["query_img"]
         
        img = Image.open(file.stream)
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        
        query = fea.extract(img)
        dists = np.linalg.norm(feature - query, axis=1)
        ids = np.argsort(dists)[:20]
        scores = [(dists[id], img_paths[id]) for id in ids] 
        
        return render_template("index.html", query_path = uploaded_img_path, scores = scores)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run()