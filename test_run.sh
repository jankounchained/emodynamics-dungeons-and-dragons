wdir="data/test_run"

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
    -d "data/sample" \
    -t "20s" \
    -o "${wdir}/bins"

python src/classifier.py \
    -d "${wdir}/bins" \
    -o "${wdir}/representations"

python src/signal_fit.py \
    -d "${wdir}/representations" \
    -w 6 \
    -t 160 \
    -o "${wdir}/results.ndjson"
