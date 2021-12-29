wdir="data/b20_w6"

# make analysis dir
if [ -d "${wdir}" ] 
then
    echo "Directory ${wdir} exists." 
else
    mkdir -p "${wdir}"
    mkdir -p "${wdir}/bins"
    mkdir -p "${wdir}/representations"
    echo "Creating ${wdir} & subdirectories"
fi

python src/binning.py \
    -d "/home/jan/D&D/data/211001_20s_sentiment/subtiltes_parsed" \
    -t "20s" \
    -o "${wdir}/bins"

python src/classifier.py \
    -d "${wdir}/bins" \
    -o "${wdir}/representations" \
    -j 1

python src/signal_fit.py \
    -d "${wdir}/representations" \
    -w 6 \
    -t 160 \
    -o "${wdir}/results.ndjson"
