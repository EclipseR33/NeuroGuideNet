import argparse
import os


def download_openbmi(save_dir=""):
    from pySmartDL import SmartDL
    from tqdm import tqdm

    save_dir = os.path.abspath(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    max_attempts = 5

    for session in range(1, 3):
        for subject in tqdm(range(1, 55)):
            filename = f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
            destination = os.path.join(save_dir, filename)
            url = (
                "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub"
                "/10.5524/100001_101000/100542/"
                f"session{session}/s{subject}/{filename}"
            )
            print(url)
            if os.path.exists(destination):
                continue

            for attempt in range(1, max_attempts + 1):
                try:
                    SmartDL(url, destination, timeout=10).start()
                    break
                except Exception as error:
                    print(f"Attempt {attempt}/{max_attempts} failed for {url}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the OpenBMI motor imagery dataset.")
    parser.add_argument(
        "--save-dir",
        default="",
        help="Directory for the downloaded files (defaults to the current working directory).",
    )
    download_openbmi(parser.parse_args().save_dir)
