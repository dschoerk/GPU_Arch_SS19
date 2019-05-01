#include "streaming_min_max_algorithms.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <getopt.h>
#include <cerrno>
#include <error.h>
#include <climits>
#include <limits>

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

static void compare_all_algos(
    std::vector<std::unique_ptr<streaming_min_max_algorithm_interface>> & algorithms,
    std::vector<float> const & data,
    std::vector<std::chrono::duration<double>> & timings,
    unsigned int window_size
    )
{
    for (unsigned int i = 0; i < algorithms.size(); ++i)
    {
	auto start = std::chrono::high_resolution_clock::now();
	algorithms[i]->calc(data, window_size);
	auto finish = std::chrono::high_resolution_clock::now();
	
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
	 << "-w, --window_size= WINDOW_SIZE[1, " << std::numeric_limits<int>::max() << "]\n"
	 << "-s, --sample_size=SAMPLE_SIZE[1, " << std::numeric_limits<int>::max() << "]\n"
	 << "-i, --iterations=ITERATIONS[1, " << std::numeric_limits<int>::max() << "]\n";

    exit(exit_code);
}

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

int main(
    int argc,
    char **argv
    )
{
    unsigned int window_size = 50;
    unsigned int sample_size = 10000;
    unsigned int number_of_iterations = 500;

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
    
    timings(algorithms, window_size, sample_size, number_of_iterations);
	    
    return EXIT_SUCCESS;
}
