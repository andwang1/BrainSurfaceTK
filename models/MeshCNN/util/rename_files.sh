# Author: Andy Wang
# License: MIT

for file in *.obj; do
    mv "$file" "${file%.obj}_R.obj"
done
