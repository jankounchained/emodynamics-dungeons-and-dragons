# folder with representations
wdir="data/b20_w6"

# window sizes to run
windows="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 30 40 50"

# run loop on bins of 20s
for w in $windows
do
outdir="data/b20_w${w}"
mkdir -p "${outdir}/results"
python src/ntr.py \
    -d "${wdir}/representations" \
    -w "${w}" \
    -o "${outdir}/results.ndjson"
done
