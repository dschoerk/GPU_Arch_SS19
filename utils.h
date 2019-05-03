#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <cstdarg>
#include <cerrno>

#ifdef __linux__
#include <error.h>
#else
extern char *program_invocation_name;

extern "C" void error_at_line(
    int status,
    int errnum,
    const char *filename,
    unsigned int linenum,
    const char *fmt,
    ...
    );
#endif


#ifdef DEBUG
    #define TRACE(format, ...)				\
        trace(						\
            "%s [%s, %s(), line %d]: " format,		\
	    program_invocation_name,			\
	    __FILE__,					\
	    __func__,					\
	    __LINE__,					\
	    ## __VA_ARGS__				\
	    )
    extern bool verbose;

    extern void trace(
        const char *fmt,
        ...
        );
#else
    #define TRACE(format, ...)
#endif


#define ERROR(format, ...)			\
    error_at_line(				\
	EXIT_SUCCESS,				\
        0,					\
        __FILE__,				\
        __LINE__,				\
        format,					\
        ## __VA_ARGS__				\
        )

#define ERROR_EXIT(format, ...)			\
    error_at_line(				\
        EXIT_FAILURE,				\
        0,					\
        __FILE__,				\
        __LINE__,				\
        format,					\
        ## __VA_ARGS__				\
        )

#define ERROR_WITH_ERRNO(format, ...)		\
    error_at_line(				\
	EXIT_SUCCESS,				\
        errno,					\
        __FILE__,				\
        __LINE__,				\
        format,					\
        ## __VA_ARGS__				\
        )

#define ERROR_EXIT_WITH_ERRNO(format, ...)	\
    error_at_line(				\
        EXIT_FAILURE,				\
        errno,					\
        __FILE__,				\
        __LINE__,				\
        format,					\
        ## __VA_ARGS__				\
        )

#endif
