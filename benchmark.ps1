$win = @(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000) # @(1024, 512, 256, 128, 64, 32, 16, 8, 4, 2)
$samples = @(1000, 10000, 100000, 1000000, 10000000)
$file = "results/results3.txt"

for ($i=0; $i -lt $win.length; $i++){
    
    $cur_win = $win[$i]
    "" >> $file
    "win: $cur_win" >> $file
    "samples, lemire, cuda_malloc, cuda_pagelocked, cuda_pagelocked_shared, thrust_naive, thrust, cuda_tiled" >> $file
    
    for ($j=0; $j -lt $samples.length; $j++){
        
        $cur_samples = $samples[$j]
        #$file = 'results/results_s{0}.txt' -f $samples[$j]
        
        .\minmax.exe -s $samples[$j] -w $win[$i] -i 11 > tmpfile.txt

        $data = Get-Content .\tmpfile.txt -Raw 

        $data -match @("lemire = ([\d\.]+) milliseconds")
        $lemire = $matches[1]

        $data -match @("cuda plain - cuda malloc = ([\d\.]+) milliseconds")
        $cuda_malloc = $matches[1]

        $data -match @("cuda plain - page locked memory = ([\d\.]+) milliseconds")
        $cuda_pagelocked = $matches[1]

        $data -match @("cuda plain - page locked shared memory = ([\d\.]+) milliseconds")
        $cuda_pagelocked_shared = $matches[1]

        $data -match @("thrust_naive = ([\d\.]+) milliseconds")
        $thrust_naive = $matches[1]

        $data -match @("thrust = ([\d\.]+) milliseconds")
        $thrust = $matches[1]

        $data -match @("cuda plain - cuda tiled = ([\d\.]+) milliseconds")
        $cuda_tiled = $matches[1]

        "$cur_samples, $lemire, $cuda_malloc, $cuda_pagelocked, $cuda_pagelocked_shared, $thrust_naive, $thrust, $cuda_tiled" >> $file
    }
}