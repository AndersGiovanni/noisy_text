for doc in submission_train test_with_noise
do
    #python preprocess/lowercase.py data/$doc.tsv data/${doc}_lower.tsv
    python preprocess/annotate_ner.py data/${doc}_lower.tsv data/${doc}_lower_entities.tsv
done
