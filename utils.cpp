#include "utils.h"
#include <cstdio>
#include <cstring>

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

#ifndef __linux__ 
void error_at_line(
    int status,
    int errnum,
    const char *filename,
    unsigned int linenum,
    const char *fmt,
    ...
    )
{
    va_list ap;
    va_start(ap, fmt);

    (void) fprintf(
	stderr,
	"%s:%s:%d: ",
	program_invocation_name,
	filename,
	linenum
	);
	
    (void) vfprintf(stderr, fmt, ap);

    va_end(ap);

    if (errnum != 0)
    {
	(void) fprintf(
	    stderr,
	    ": %s",
	    strerror(errnum)
	    );
    }

    (void) fputs("\n", stderr);
		
    if (status != EXIT_SUCCESS)
    {
	exit(status);
    }
}
#endif
