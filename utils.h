#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <cstdarg>

/// - can be removed?
#ifdef __linux__ 
    #include <getopt.h>
#endif
///

#include <cerrno>
    

#ifdef __linux__ 
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
    #endif
#else
    #define ERROR(format, ...) /*\
        printf(				\
        format,					\
            ## __VA_ARGS__				\
        ); exit(1)*/


    #define ERROR_EXIT(format, ...)
    #define TRACE(format, ...)
#endif
#endif