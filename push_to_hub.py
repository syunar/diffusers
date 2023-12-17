from datasets import load_dataset, load_from_disk

def main():
    ds = load_from_disk("/home/users/fronk/projects/deepcolor/datasets/greyscale_manga_fullsize_with_caption")
    succesful_competion = False
    while not succesful_competion:
        try:
            ds.push_to_hub('dsupa/greyscale_manga_fullsize_with_caption_550k', private=True, max_shard_size="1GB")
            succesful_competion = True
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()