for file in *.obj; do
    mv "$file" "${file%.obj}_L.obj"
done
