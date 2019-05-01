#include "streaming_min_max_lemire.h"

#include <deque>

std::string_view streaming_min_max_lemire::get_name(
    ) const
{
    return "lemire";
}
  
void streaming_min_max_lemire::calc(
    std::vector<float> const & array,
    unsigned int width
    )
{
    maxvalues.clear();
    maxvalues.resize(array.size() - width + 1);
    minvalues.clear();
    minvalues.resize(array.size() - width + 1);

    std::deque<int> maxfifo, minfifo;

    for (unsigned int i = 1; i < width; ++i) {
	if (array[i] > array[i - 1]) { // overshoot
	    minfifo.push_back(i - 1);
	    while (!maxfifo.empty()) {
		if (array[i] <= array[maxfifo.back()]) {
		    if (i == width + maxfifo.front())
			maxfifo.pop_front();
		    break;
		}
		maxfifo.pop_back();
	    }
	} else {
	    maxfifo.push_back(i - 1);
	    while (!minfifo.empty()) {
		if (array[i] >= array[minfifo.back()]) {
		    if (i == width + minfifo.front())
			minfifo.pop_front();
		    break;
		}
		minfifo.pop_back();
	    }
	}
    }
    for (unsigned int i = width; i < array.size(); ++i) {
	maxvalues[i - width] =
	    array[maxfifo.empty() ? i - 1 : maxfifo.front()];
	minvalues[i - width] =
	    array[minfifo.empty() ? i - 1 : minfifo.front()];
	if (array[i] > array[i - 1]) { // overshoot
	    minfifo.push_back(i - 1);
	    if (i == width + minfifo.front())
		minfifo.pop_front();
	    while (!maxfifo.empty()) {
		if (array[i] <= array[maxfifo.back()]) {
		    if (i == width + maxfifo.front())
			maxfifo.pop_front();
		    break;
		}
		maxfifo.pop_back();
	    }
	} else {
	    maxfifo.push_back(i - 1);
	    if (i == width + maxfifo.front())
		maxfifo.pop_front();
	    while (!minfifo.empty()) {
		if (array[i] >= array[minfifo.back()]) {
		    if (i == width + minfifo.front())
			minfifo.pop_front();
		    break;
		}
		minfifo.pop_back();
	    }
	}
    }
    maxvalues[array.size() - width] =
	array[maxfifo.empty() ? array.size() - 1 : maxfifo.front()];
    minvalues[array.size() - width] =
	array[minfifo.empty() ? array.size() - 1 : minfifo.front()];
}

std::vector<float> const & streaming_min_max_lemire::get_max_values(
    ) const
{
    return maxvalues;
}

std::vector<float> const & streaming_min_max_lemire::get_min_values(
    ) const
{
    return minvalues;
}
