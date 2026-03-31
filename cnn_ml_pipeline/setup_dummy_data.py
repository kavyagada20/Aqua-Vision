import os
from PIL import Image

def generate_dummy_data(base_path, classes, num_images_per_class=10):
    os.makedirs(base_path, exist_ok=True)
    
    colors = {
        classes[0]: (135, 206, 235), # Sky blue (representing clean water)
        classes[1]: (139, 69, 19)   # Saddle brown (representing polluted water)
    }

    for cls in classes:
        class_path = os.path.join(base_path, cls)
        os.makedirs(class_path, exist_ok=True)
        
        for i in range(num_images_per_class):
            img = Image.new('RGB', (224, 224), color=colors[cls])
            img.save(os.path.join(class_path, f"{cls}_{i}.jpg"))
            
    print(f"Generated {num_images_per_class} images for classes: {classes} in {base_path}")

if __name__ == "__main__":
    generate_dummy_data('dataset', ['clean_water', 'polluted_water'], num_images_per_class=20)
