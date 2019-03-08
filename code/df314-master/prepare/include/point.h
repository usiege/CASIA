#ifndef __POINT_H__
#define __POINT_H__

#include <cmath>
#include <cstdio>
#include <stdint.h>

struct Point
{
    Point()
        : x(0), y(0), z(0)
    { }

    Point(float _x, float _y, float _z)
        : x(_x), y(_y), z(_z)
    { }

    float x;
    float y;
    float z;
};

inline Point atopoint(const char* buffer)
{
    Point p;
    sscanf(buffer, "%f,%f,%f", &p.x, &p.y, &p.z);
    return p;
}

#endif
