echo "Compiling GPU_Kmeans..."
make all
echo "Running GPU_python.py..."
python3 comparisons/GPU_python.py -d $2 -k $3 -m $4 -f $5 -o "out/GPU_python/$(basename $5)"
echo "Running GPU_Kmeans..."
./bin/src/main -n $1 -d $2 -k $3 -m $4 -o "out/GPU_Kmeans/$(basename $5)" < $5
echo "Showing plots..."
python3 py_utils/scatter_plot.py -f "out/GPU_Kmeans/$(basename $5)" "out/GPU_python/$(basename $5)" -d $2
echo "DONE!"