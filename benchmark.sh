win="100 200 300 400 500 600 700 800 900 1000" # @(1024, 512, 256, 128, 64, 32, 16, 8, 4, 2)
samples="1000 10000 100000 1000000 10000000"
file="results/results_herz.txt"
executable="./streaming_min_max_comparison"

date > ${file}

for cur_win in ${win}
do    
    echo "win: ${cur_win}" >> ${file}
    echo "samples, lemire, cuda_malloc, cuda_pagelocked, cuda_pagelocked_shared, thrust_naive, thrust, cuda_tiled" >> ${file}
    
    for cur_sample in ${samples}
    do
	echo -n ${cur_sample} >> ${file}

        for timing in `${executable} -s ${cur_sample} -w ${cur_win} -i 11 | grep milliseconds | cut -d '=' -f 2 | cut -d ' ' -f 2`
	do
	    echo -n ", ${timing}" >> ${file}
	done

	echo >> ${file}
    done
done
