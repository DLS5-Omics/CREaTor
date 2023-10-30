# CREaTor

Cross-cell type gene expression prediction with deep learning reveals systematic cis-regulatory patterns at hierarchical levels

## Getting Started

### Installation

1. Clone the repo

```sh
git clone https://github.com/DLS5-Omics/CREaTor.git
```

2. Install python dependencies

```sh
pip install -r requirements.txt
```

## Usage

```sh
python CREaTor.py -i <INPUT>
```

## Example

```sh
cd example
./run_example.sh
```

## Citation

```bibtex
@article {li2023modeling,
	author = {Yongge Li and Fusong Ju and Zhiyuan Chen and Yiming Qu and Huanhuan Xia and Liang He and Lijun Wu and Jianwei Zhu and Bin Shao and Pan Deng},
	title = {CREaTor: zero-shot cis-regulatory pattern modeling with attention mechanisms},
	elocation-id = {2023.03.28.534267},
	year = {2023},
	doi = {10.1101/2023.03.28.534267},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Linking cis-regulatory sequences to target genes has been a long-standing challenge due to the intricate nature of gene regulation. Here, we present a hierarchical deep neural network, CREaTor, to decode cis-regulatory mechanisms across cell types by predicting gene expression from flanking candidate cis-regulatory elements (cCREs). With attention mechanism as the core component in our network, we can model complex interactions between genomic elements as far as 2Mb apart. This allows a more accurate and comprehensive depiction of gene regulation that involves cis-regulatory programs. Testing with a held-out cell type demonstrates that CREaTor outperforms previous methods in capturing cCRE-gene interactions spanning varying distance ranges. Further analysis suggests that the performance of CREaTor may be attributed to its ability to model regulatory interactions at multiple levels, including higher-order genome organizations that govern cCRE activities and cCRE-gene interactions. Together, this study showcases CREaTor as a powerful tool for systematic study of cis-regulatory programs in different cell types involved in normal developmental processes and diseases.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2023/03/29/2023.03.28.534267},
	eprint = {https://www.biorxiv.org/content/early/2023/03/29/2023.03.28.534267.full.pdf},
	journal = {bioRxiv}
}
```

## License

See `LICENSE` for more information.

`SPDX-License-Identifier: GPL-3.0-or-later`
