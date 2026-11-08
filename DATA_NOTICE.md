# Data notice

The three `.npz` files in `data/` are frozen caches derived from the FEMNIST dataset distributed through the LEAF federated-learning benchmark. FEMNIST is based on handwritten-character data from NIST Special Database 19.

- `femnist_evaluation.npz`: 8,064 training examples and 1,021 evaluation examples.
- `femnist_calibration.npz`: the same training arrays and 995 disjoint calibration examples.
- `femnist_original_heldout.npz`: the held-out pool used to verify that calibration and evaluation are disjoint.

Each cache stores:

- `x_train`, `y_train`, `writer_train`;
- `x_test`, `y_test`, `writer_test`.

Images are `float32` values in `[0,1]`, stored as flattened 14-by-14 arrays. Writer identifiers are contiguous integers from 0 through 59, and labels are integers from 0 through 61.

The caches are included because they are required to reproduce the exact experimental protocol. The package does not reconstruct the earlier raw FEMNIST download, writer selection, or image-resizing pipeline.

Please cite:

S. Caldas, S. M. K. Duddu, P. Wu, T. Li, J. Konečný, H. B. McMahan, V. Smith, and A. Talwalkar, “LEAF: A Benchmark for Federated Settings,” 2018.

Upstream resources:

- LEAF repository: `https://github.com/TalwalkarLab/leaf`
- NIST Special Database 19: `https://www.nist.gov/srd/nist-special-database-19`

## Redistribution and attribution

The accompanying `LICENSE` covers the software authored for this repository, not a blanket grant of rights over third-party datasets. These are transformed FEMNIST/LEAF caches based on NIST Special Database 19. Please check applicable upstream terms before further redistribution and acknowledge NIST and LEAF when using these data. NIST publishes guidance on data licensing at https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications .

The source pipeline for selecting writers and resizing images is not present, and an exact reconstruction from raw upstream data is therefore not claimed.
