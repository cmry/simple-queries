
FTDIR=$(pwd)/fastText
RESULTDIR=$(pwd)/scores
DATADIR=$(pwd)/data

DIM=30
LR=0.1
WORDGRAMS=2
MINCOUNT=2
MINN=3
MAXN=3
BUCKET=1000000
EPOCH=20
THREAD=20

echo "DIM: $DIM - LR: $LR - (WGRAMS: $WORDGRAMS - MINC: $MINCOUNT) - CHARS ($MINN, $MAXN) - EPS: $EPOCH \n\n"

$FTDIR/fasttext supervised -input "${DATADIR}/$1.train" -output "${RESULTDIR}/$1" -dim $DIM -lr $LR -wordNgrams $WORDGRAMS -minCount $MINCOUNT -minn $MINN -maxn $MAXN -bucket $BUCKET -epoch $EPOCH -thread $THREAD

$FTDIR/fasttext test "${RESULTDIR}/$1.bin" "${DATADIR}/$2.test"
