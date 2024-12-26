import os
import shutil
import argparse
import awkward as ak


def read_and_group_files(directory):
    multijet_files = []
    ggF_k01_files = []
    ggF_k05_files = []

    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            filepath = os.path.join(directory, filename)
            if "multijet" in filename:
                multijet_files.append(filepath)
            elif "ggF_k01" in filename:
                ggF_k01_files.append(filepath)
            elif "ggF_k05" in filename:
                ggF_k05_files.append(filepath)

    return multijet_files, ggF_k01_files, ggF_k05_files


def concatenate_and_save(files, output_filename):
    if not files:
        print(f"No files found for {output_filename}. Skipping.")
        return
    try:
        concatenated_array = ak.concatenate([ak.from_parquet(file) for file in files])
        mask = (
            concatenated_array.signal_4btags_GN2v01_77_min_mass_optimized_1D_medium_pairing_mask
        )
        print(
            output_filename,
            "min_mass_optimized_1D_medium_pairing",
            ak.sum(concatenated_array.event_weight[mask]),
        )
        mask = concatenated_array.signal_4btags_GN2v01_77_min_deltar_pairing_mask
        print(
            output_filename,
            "min_deltar_pairing",
            ak.sum(concatenated_array.event_weight[mask]),
        )
        # ak.to_parquet(concatenated_array, output_filename)
        # print(f"Saved concatenated file: {output_filename}")
    except Exception as e:
        print(f"Error processing {output_filename}: {e}")


def main(directory):
    multijet_files, ggF_k01_files, ggF_k05_files = read_and_group_files(directory)
    concatenate_and_save(ggF_k01_files, "ggF_k01_concatenated.parquet")
    # concatenate_and_save(ggF_k05_files, "ggF_k05_concatenated")
    # concatenate_and_save(multijet_files, "multijet_concatenated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate Parquet files into groups and save them."
    )
    parser.add_argument(
        "directory", type=str, help="Path to the directory containing Parquet files"
    )
    args = parser.parse_args()

    main(args.directory)
