# Metastasis-glmGCN
Presentation
Meta-GDBP is a high-level stacked algorithm which has two stages. At the top-level stage, It contains four base models (BMs)--GBM,DBM,PBM,BBM.
We use each BM to predict response of the drug-cell line pair in CCLE and CGP.
At the bottom-level stage, We use the weighted method to predict response of the drug-cell line pair in CCLE and CGP.

R version
R-3.4.2

How to Use
First, you can download the code and put them in your R workspace.
Then, you can use the following sequence to read the code:pearson->fingerprint->wgcna->kegg->go->cc_elasticnet->cc_svr->dd_fp->kegg_normal->go_normal->lm
We have put serveral example input files and output files in each dir. You can use them as reference. (The pearson,wgcna,kegg,go dirs need exprSet data.You need to put it in the dir. The file is too large to push it.)
