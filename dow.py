import os
import requests
import time

SAVE_DIR = "case10_all_images_all_slices"

MODALITIES = {
    "MR-T2": "https://www.med.harvard.edu/aanlib/cases/caseNN1/mr1/{num:03d}.png",
    "MR-T1": "https://www.med.harvard.edu/aanlib/cases/caseNN1/mr2/{num:03d}.png",
    "PET": "https://www.med.harvard.edu/aanlib/cases/caseNN1/dg1/{num:03d}.png",
}

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

# chỉ bỏ icon/logo/button
SKIP_WORDS = [
    "icon",
    "logo",
    "button",
    "arrow",
    "home",
    "help",
    "next",
    "prev",
    "blank"
]

def should_skip(url):
    lower = url.lower()
    return any(word in lower for word in SKIP_WORDS)

for modality, pattern in MODALITIES.items():
    folder = os.path.join(SAVE_DIR, modality)
    os.makedirs(folder, exist_ok=True)

    print(f"\nDownloading {modality}...")

    for num in range(1, 200):
        url = pattern.format(num=num)

        if should_skip(url):
            print("Skip icon:", url)
            continue

        filename = f"{num:03d}.png"
        save_path = os.path.join(folder, filename)

        try:
            r = session.get(url, timeout=10)

            # chỉ cần tồn tại là tải
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)

                print("Downloaded:", modality, filename)

            else:
                print("Not found:", modality, filename)

        except Exception as e:
            print("Fail:", modality, filename, e)

        time.sleep(0.2)

print("\nDONE")
print("Saved in:", SAVE_DIR)