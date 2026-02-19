import albumentations as A
import cv2
import os

# ==============================
# SETTINGS
# ==============================
INPUT_DIR = "data"        # dataset/A , dataset/B ...
OUTPUT_DIR = "aug_dataset"
TARGET_PER_CLASS = 600

# ==============================
# AUGMENTATION PIPELINE
# ==============================
transform = A.Compose([
    A.HorizontalFlip(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.12,
        scale_limit=0.15,
        rotate_limit=20,
        shear=10,
        p=0.8
    ),

    A.RandomBrightnessContrast(p=0.7),
    A.GaussianBlur(blur_limit=(3,5), p=0.2),
    A.GaussNoise(p=0.2)
])

# ==============================
# CREATE OUTPUT DIR
# ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# PROCESS EACH CLASS
# ==============================
for class_name in os.listdir(INPUT_DIR):

    class_path = os.path.join(INPUT_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    out_class = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(out_class, exist_ok=True)

    images = [img for img in os.listdir(class_path) if img.lower().endswith((".jpg",".png",".jpeg"))]

    original_count = len(images)

    if original_count == 0:
        continue

    aug_per_image = max(1, TARGET_PER_CLASS // original_count)

    print(f"{class_name}: original={original_count}, aug_per_image={aug_per_image}")

    count = 0

    for img_name in images:

        img_path = os.path.join(class_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        # save original
        cv2.imwrite(os.path.join(out_class, f"orig_{img_name}"), image)
        count += 1

        # create augmented images
        for i in range(aug_per_image):

            augmented = transform(image=image)
            aug_img = augmented["image"]

            new_name = f"aug_{count}_{img_name}"
            cv2.imwrite(os.path.join(out_class, new_name), aug_img)

            count += 1

            if count >= TARGET_PER_CLASS:
                break

        if count >= TARGET_PER_CLASS:
            break

print("\n✅ Augmentation finished")
