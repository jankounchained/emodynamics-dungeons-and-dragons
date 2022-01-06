# window sizes to run
bins="5 10 15 20 25 30 40"

# run loop on bins of 20s
for bin in $bins
do
outdir="data/b${bin}_w6"
mkdir -p "${outdir}"
mkdir -p "${outdir}/bins"
mkdir -p "${outdir}/representations"
python src/binning.py \
    -d "/home/jan/D&D/data/211001_20s_sentiment/subtiltes_parsed" \
    -t "${bin}s" \
    -o "${outdir}/bins"

python src/classifier.py \
    -d "${outdir}/bins" \
    -o "${outdir}/representations" \
    -j 1

python src/ntr.py \
    -d "${outdir}/representations" \
    -w 6 \
    -o "${outdir}/results.ndjson"
done
