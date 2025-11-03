import os

def get_folders(dataset):
    root_dir = os.path.join(r"C:\Users\thiag\DatasetsCv2", dataset)

    output_file = "paths.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for filename in os.listdir(root_dir):
            path = os.path.join(root_dir, filename)
            if os.path.isdir(path):
                f.write(path + "\n")

    print(f"Paths salvos em {output_file}")
