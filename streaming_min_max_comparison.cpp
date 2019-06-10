#include "streaming_min_max_algorithms.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <iostream>

#ifdef __linux__ 
    #include <getopt.h>
    #include <error.h>
#endif

#include <cerrno>
#include <climits>
#include <limits>
#include <algorithm>

#ifndef __linux__ 
char *program_invocation_name("<not yet set>");
#endif

#ifdef DEBUG
bool verbose(false);
#endif // DEBUG

static std::vector<float> create_white_noise_vector(
    unsigned int size
    )
{
    std::vector<float> data(size);

    for (unsigned int k = 0; k < size; ++k)
    {
        data[k] = (1.0 * rand() / (RAND_MAX)) - 0.5;
    }
    
    return data;
}

static bool check_for_difference(
    std::string_view minmax,
    std::string_view reference_name,
    std::vector<float> const & reference_values,
    std::string_view compare_name,
    std::vector<float> const & compare_values
    )
{
    if (reference_values.size() != compare_values.size())
    {
	ERROR(
	    "Number of %s values differ between %s and %s "
	    "(%lu vs. %lu)",
	    minmax.data(),
	    reference_name.data(),
	    compare_name.data(),
	    reference_values.size(),
	    compare_values.size()
	    );

	return true;
    }

    auto mismatch =
	std::mismatch(
	    reference_values.begin(),
	    reference_values.end(),
	    compare_values.begin()
	    );
    bool retval = false;

    if (mismatch.first != reference_values.end())
    {
	ERROR(
	    "Some %s values differ between %s and %s",
	    minmax.data(),
	    reference_name.data(),
	    compare_name.data()
	    );
    }
		
    while (mismatch.first != reference_values.end())
    {
	auto position = mismatch.first - reference_values.begin();
	retval = true;
	
	std::cerr << "\t" << position << ": " << *(mismatch.first)
		  << " != " << *(mismatch.second) << '\n';

	mismatch =
	    std::mismatch(
		mismatch.first + 1,
		reference_values.end(),
		mismatch.second + 1
		);
    }

    return retval;
}

static void compare_all_algos(
    std::vector<std::unique_ptr<streaming_min_max_algorithm_interface>> & algorithms,
    std::vector<float> const & data,
    std::vector<std::chrono::duration<double>> & timings,
    unsigned int window_size
    )
{
    

    for (unsigned int i = 0; i < algorithms.size(); ++i)
    {
        auto reference_name = algorithms[0]->get_name();
        auto start = std::chrono::high_resolution_clock::now();
        algorithms[i]->calc(data, window_size);
        auto finish = std::chrono::high_resolution_clock::now();

	//
	// store results of Lemire's algorithms as a reference for
	// a diff-check
	//
        auto& reference_max_values = algorithms[0]->get_max_values();
        auto& reference_min_values = algorithms[0]->get_min_values();
	    

        /*printf("Ref Values\n");
        for(auto v : reference_min_values) 
            printf("%.2f, ", v);
        printf("\n");*/


        if (algorithms[i]->check_against_reference())
        {
            auto& compare_max_values = algorithms[i]->get_max_values();
            auto& compare_min_values = algorithms[i]->get_min_values();
            auto compare_name = algorithms[i]->get_name();

            bool difference_min = check_for_difference(
                "min",
                reference_name,
                reference_min_values,
                compare_name,
                compare_min_values
            );

            /*printf("%s Values\n", std::string(compare_name).c_str());
            for(auto v : compare_min_values) 
                printf("%.2f, ", v);
            printf("\n");*/

            bool difference_max = check_for_difference(
                "max",
                reference_name,
                reference_max_values,
                compare_name,
                compare_max_values
            );
        }
        
        timings[i] += (finish - start);
    }
}

static void timings(
    std::vector<std::unique_ptr<streaming_min_max_algorithm_interface>> & algorithms,
    unsigned int window_size,
    unsigned int sample_size,
    unsigned int number_of_iterations
    )
{
    std::vector<std::chrono::duration<double>> timings(algorithms.size());

    std::cout << "\nPerforming a comparison using the following parameters:"
        << "\n  window_size = " << window_size
        << "\n  sample_size = " << sample_size
        << "\n  number_of_iterations = " << number_of_iterations
        << "\n\n";

    for (unsigned int i = 0; i < number_of_iterations; ++i)
    {
        auto data = create_white_noise_vector(sample_size);
        
	TRACE("Input sample contains the following %d values:\n");

#ifdef DEBUG
	if (verbose)
	{
	    for (unsigned int j = 0; j < data.size(); ++j)
	    {
		std::cout << '\t' << j << ": " << data[j] << '\n';
	    }
	}
#endif

        compare_all_algos(algorithms, data, timings, window_size);
    }

    for (unsigned int i = 0; i < algorithms.size(); ++i)
    {
	    std::cout << algorithms[i]->get_name() << " = " << timings[i].count() << " seconds\n";
    }
}

/**
 * \brief Print a usage message
 *
 * Prints a usage message to \a stream and terminate program with \a exit_code
 *
 * \param fp stdio stream to write usage message to [IN]
 * \param exit_code exit code returned to the environment [IN]
 */
static void usage(
    std::ostream &ostr,
    int exit_code
    )
{
    size_t i;

    ostr << "Usage: " << program_invocation_name << " [OPTIONS]\n\n"
	 << "where OPTIONS may be one or multiple of the following:\n"
	 << "-h, --help\n"
#ifdef DEBUG
	 << "-v, --verbose\n"
#endif // DEBUG      
	 << "-w, --window_size=WINDOW_SIZE[1, " << std::numeric_limits<int>::max() << "]\n"
	 << "-s, --sample_size=SAMPLE_SIZE[1, " << std::numeric_limits<int>::max() << "]\n"
	 << "-i, --iterations=ITERATIONS[1, " << std::numeric_limits<int>::max() << "]\n";

    exit(exit_code);
}

#ifdef __linux__
/**
 * \brief Converts a string into a unsigned int
 *
 * Converts a string into a unsigned int doing overflow and underflow checking.
 * Terminates with a \a usage() message in case of an error
 *
 * \param str string to covert [IN]
 * \return numerical value corresponding to string
 */
static unsigned int checked_to_uint(
    const char *str
    )
{
    char *end_ptr = NULL;
    long const result = strtol(str, &end_ptr, 10);

    if (((result == LONG_MIN) || (result == LONG_MAX)) && (errno == ERANGE))
    {
	usage(std::cerr, EXIT_FAILURE);
    }

    if ((end_ptr == NULL) || (end_ptr == str) || (*end_ptr != '\0'))
    {
	usage(std::cerr, EXIT_FAILURE);
    }

    if ((result < 1) || (result > std::numeric_limits<int>::max()))
    {
	usage(std::cerr, EXIT_FAILURE);
    }

    return static_cast<unsigned int>(result);
}
#endif

int main(
    int argc,
    char **argv
    )
{
    unsigned int window_size = 50;
    unsigned int sample_size = 10000000;
    unsigned int number_of_iterations = 1;

#ifndef __linux__ 
    program_invocation_name = argv[0];
#endif

#ifdef __linux__
    char c;
    struct option long_options[] =
    {
        {"help", no_argument, NULL, 'h'},
#ifdef DEBUG
        {"verbose", no_argument, NULL, 'v'},
#endif // DEBUG	
        {"window_size", required_argument, NULL, 'w'},
        {"sample_size", required_argument, NULL, 's'},
        {"iterations", required_argument, NULL, 'i'},
        {0, 0, 0, 0}
    };

    while (
	(c = getopt_long(
	    argc,
	    argv,
#ifdef DEBUG
	    "v"
#endif // DEBUG	
	    "hw:s:i:",
	    long_options,
	    NULL)
	    ) != -1
	)
    {
        switch (c)
        {
            case 'h':
                usage(std::cout, EXIT_SUCCESS);
                break;
#ifdef DEBUG
            case 'v':
	        verbose = true;
                break;
#endif // DEBUG	
            case 'w':
                window_size = checked_to_uint(optarg);
                break;
            case 's':
                sample_size = checked_to_uint(optarg);
                break;
            case 'i':
                number_of_iterations = checked_to_uint(optarg);
                break;
            case '?':
            default:
                usage(std::cerr, EXIT_FAILURE);
                break;
        }
    }

    if ((optind < argc))
    {
        usage(std::cerr, EXIT_FAILURE);
    }
#endif // __ linux __
    
    timings(algorithms, window_size, sample_size, number_of_iterations);
	    
    return EXIT_SUCCESS;
}
