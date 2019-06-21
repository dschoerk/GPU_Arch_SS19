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

const unsigned int colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const size_t num_colors = sizeof(colors)/sizeof(int);

#define USE_NVTX
#ifdef USE_NVTX
#include "nvToolsExt.h"
#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
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
