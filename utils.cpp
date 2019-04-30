#include "utils.h"
#include <cstdio>

#ifdef DEBUG
void trace(
    const char *fmt,
    ...
    )
{
    if (verbose)
    {
        va_list ap;
        va_start(ap, fmt);

        if (vfprintf(stdout, fmt, ap) < 0)
        {
            ERROR("Cannot write to trace stream");
        }
	
        va_end(ap);

	if (fflush(stdout) == EOF)
	{
	    ERROR("Cannot flush trace stream");
	}
    }
}
#endif // DEBUG
