# Segmentácia jadier

**Autor: Bc. Ivan Vykopal**

Tento adresár obsahuje experimenty vykonané pre segmentáciu jadier v histologických snímkach.

Pri trénovaní využívame dva datasety a to Lizard a MoNuSeg datasety, ktoré sme si prispôsobili pre binárnu segmentáciu jadier.

Obsah adresára:

- `Binary_mask_generation.ipynb` -  Notebook poskytnutý spolu s dátami pre MoNuSac dataset.
- `Lizard create masks.ipynb` - Notebook pre vytvorenie binárnych masiek jadier.
- `Merger patches.ipynb` - Notebook pre spojene vytvorených predikcií do pôvodnej veľkosti.
- `Preprocessing.ipynb` - Notebook pre vytvorenie neprekrývajúcich sa výsekov z histologických snímok.
- `U-Net.ipynb` - Notebook určený pre trénovanie U-Net modelu pre segmentáciu jadier z Lizard alebo MoNuSeg datasetu.
- `U-Net++.ipynb` - Notebook určený pre trénovanie U-Net++ pre segmentáciu jadier z Lizard alebo MoNuSeg datasetu.
