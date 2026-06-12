import os
from collections import Counter
from PIL import Image

DATA_DIR = "data/skin_cancer"

def analyze_split(split_path):
    info = {
        "total_images": 0,
        "class_counts": Counter(),
        "modes": Counter(),
        "sizes": [],
        "corrupted": []
    }

    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)

            try:
                with Image.open(file_path) as img:
                    info["total_images"] += 1
                    info["class_counts"][class_name] += 1
                    info["modes"][img.mode] += 1
                    info["sizes"].append(img.size)   # (width, height)
            except Exception:
                info["corrupted"].append(file_path)

    return info


def summarize(name, info):
    print(f"\n===== {name.upper()} =====")
    print("Total images:", info["total_images"])
    print("Class counts:", dict(info["class_counts"]))
    print("Color modes:", dict(info["modes"]))
    print("Corrupted images:", len(info["corrupted"]))

    if info["sizes"]:
        widths = [s[0] for s in info["sizes"]]
        heights = [s[1] for s in info["sizes"]]

        print("Min width:", min(widths), "| Max width:", max(widths))
        print("Min height:", min(heights), "| Max height:", max(heights))
        print("Average width:", round(sum(widths) / len(widths), 2))
        print("Average height:", round(sum(heights) / len(heights), 2))

        unique_sizes = Counter(info["sizes"])
        print("\nTop 10 most common image sizes:")
        for size, count in unique_sizes.most_common(10):
            print(f"  {size}: {count}")


def main():
    train_path = os.path.join(DATA_DIR, "train")
    test_path = os.path.join(DATA_DIR, "test")

    train_info = analyze_split(train_path)
    test_info = analyze_split(test_path)

    summarize("train", train_info)
    summarize("test", test_info)

    if train_info["corrupted"] or test_info["corrupted"]:
        print("\nSome corrupted images found:")
        for p in train_info["corrupted"][:10] + test_info["corrupted"][:10]:
            print(p)


if __name__ == "__main__":
    main()