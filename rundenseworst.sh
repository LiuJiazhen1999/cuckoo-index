dir="denseworstdata/" #需要运行的csv文件在的位置，需要在CUCKOO-INDEX目录下新建一个wikipedia文件夹，然后把所有的csv mv过来
column="lon" #需要测试的列名
logpath="/searchlogs/denseworst.workload" #查询负载的路径
ls $dir | while read line
do
    file=${dir}${line}
    echo "$file start"
    sed -i "31c\        \"$file\"" BUILD.bazel
    sed -i "378c\        \"$file\"" BUILD.bazel
    sed -i "280c\  std::ifstream file(\"$(pwd)$logpath\");" lookup_benchmark.cc
    bazel run -c opt --cxxopt='-std=c++17' --dynamic_mode=off :lookup_benchmark -- --input_csv_path="$file" --columns_to_test="$column" > denseworstlog/$line.log #>后面是表示运行日志写入的地方
    echo "$file end"
done