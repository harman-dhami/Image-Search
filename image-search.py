from PIL import Image
from pathlib import Path
import numpy as np

from features import Features

if __name__ == "__main__":
    
    fea = Features()
    
    for img_path in sorted (Path("static/random_images").glob("*.jpg")):
        print(img_path)
        
        feature = fea.extract(img=Image.open(img_path))
        print(type(feature), feature.shape)
        
        feature_path = Path("static/feature") / (img_path.stem + ".npy")
        print(feature_path)
        
        np.save(feature_path, feature)