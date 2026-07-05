#pragma once

#include <cstdlib>
#include <cstring>
#include <limits>

#include "superlu_ddefs.h"

static inline int superlu_env_truthy(const char *value)
{
    if (value == NULL || value[0] == '\0') return -1;
    if (!std::strcmp(value, "1") || !std::strcmp(value, "true") ||
        !std::strcmp(value, "TRUE") || !std::strcmp(value, "yes") ||
        !std::strcmp(value, "YES") || !std::strcmp(value, "on") ||
        !std::strcmp(value, "ON"))
        return 1;
    if (!std::strcmp(value, "0") || !std::strcmp(value, "false") ||
        !std::strcmp(value, "FALSE") || !std::strcmp(value, "no") ||
        !std::strcmp(value, "NO") || !std::strcmp(value, "off") ||
        !std::strcmp(value, "OFF"))
        return 0;
    return -1;
}

static inline bool superlu_sym_v2_env_bool_flag(
    const char *name, int fallback)
{
    const char *env = std::getenv(name);
    if (env == NULL || env[0] == '\0') return fallback != 0;

    const int parsed = superlu_env_truthy(env);
    if (parsed < 0) ABORT("Invalid boolean GPU3DV2 environment value.");
    return parsed != 0;
}

static inline size_t superlu_sym_v2_env_size_flag(
    const char *name, size_t fallback, size_t min_value)
{
    const char *env = std::getenv(name);
    if (env == NULL || env[0] == '\0') return fallback;

    char *end = NULL;
    unsigned long long value = std::strtoull(env, &end, 10);
    if (end == env || *end != '\0' ||
        value < static_cast<unsigned long long>(min_value) ||
        value > static_cast<unsigned long long>(
            std::numeric_limits<size_t>::max()))
        ABORT("Invalid integer GPU3DV2 environment value.");
    return static_cast<size_t>(value);
}
