#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <cstdarg>
#include <getopt.h>
#include <cerrno>
#include <error.h>

#define ERROR(format, ...)			\
    error_at_line(				\
        EXIT_SUCCESS,				\
	errno,					\
	__FILE__,				\
	__LINE__,				\
	format,					\
        ## __VA_ARGS__				\
	)

#define ERROR_EXIT(format, ...)			\
    error_at_line(				\
        EXIT_FAILURE,				\
	errno,					\
	__FILE__,				\
	__LINE__,				\
	format,					\
        ## __VA_ARGS__				\
	)

#ifdef DEBUG
#define TRACE(format, ...)			\
    trace(					\
        "%s [%s, %s(), line %d]: " format,	\
        program_invocation_name,		\
        __FILE__,				\
        __func__,				\
        __LINE__,				\
        ## __VA_ARGS__				\
        )
extern bool verbose;

extern void trace(
    const char *fmt,
    ...
    );
#else
#define TRACE(format, ...)
#endif // DEBUG

#endif
