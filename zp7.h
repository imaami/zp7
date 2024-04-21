#ifndef ZP7_H_
#define ZP7_H_

#ifndef __cplusplus
#include <stdint.h>
#define STD
#else // __cplusplus
#include <cstdint>
#define STD ::std::
extern "C" {
#endif // __cplusplus

extern STD uint32_t _pext_u32(STD uint32_t a, STD uint32_t mask);
extern STD uint64_t _pext_u64(STD uint64_t a, STD uint64_t mask);
extern STD uint32_t _pdep_u32(STD uint32_t a, STD uint32_t mask);
extern STD uint64_t _pdep_u64(STD uint64_t a, STD uint64_t mask);

#undef STD

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // ZP7_H_
